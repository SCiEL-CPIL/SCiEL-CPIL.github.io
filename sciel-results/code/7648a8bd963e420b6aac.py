import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Constants for time management
DEADLINE = 330.0  # Seconds
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    if np.any(dists[np.triu_indices(n, k=1)] < min_dist):
        return False
    return True

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    a = 1.0
    R_est = (3 * n * a**3 / (16 * np.pi)) ** (1/3) * 1.5
    max_idx = int(R_est / a) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                base = np.array([i, j, k]) * a
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5, 0.5]) * a,
                    np.array([0.5, 0, 0.5]) * a,
                    np.array([0.5, 0.5, 0]) * a
                ]
                for off in offsets:
                    p = base + off
                    if np.linalg.norm(p) <= R_est:
                        points.append(p)
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idxs = np.argsort(dists)[:n]
    return points[idxs]

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    a = 1.0
    R_est = (3 * n * a**3 / (8 * np.pi)) ** (1/3) * 1.5
    max_idx = int(R_est / a) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                base = np.array([i, j, k]) * a
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0.5, 0.5, 0.5]) * a
                ]
                for off in offsets:
                    p = base + off
                    if np.linalg.norm(p) <= R_est:
                        points.append(p)
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idxs = np.argsort(dists)[:n]
    return points[idxs]

def _generate_icosahedron(n):
    """Generate positions for a Mackay icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    points = [np.zeros(3)]
    layer_dist = 1.0
    points.extend([v * layer_dist for v in verts])
    # Add edge midpoints for layer 2
    for i in range(12):
        for j in range(i+1, 12):
            v1, v2 = verts[i], verts[j]
            if np.linalg.norm(v1 - v2) < 1.1:
                mid = (v1 + v2) / 2
                mid = mid / np.linalg.norm(mid)
                points.append(mid * (np.sqrt(2)))
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idxs = np.argsort(dists)[:n]
    return points[idxs]

def _growth_algorithm(n, kind, potential, max_time_per_step=15.0):
    """
    Sequential growth algorithm: build cluster by adding atoms to lowest-energy
    interstitial sites of a relaxed (N-1)-atom cluster.
    """
    # Start with a small seed cluster
    seed_n = min(12, n)
    seed_pos = _generate_icosahedron(seed_n)
    if seed_pos is None:
        seed_pos = _generate_fcc(seed_n)
    if seed_pos is None:
        seed_pos = np.array([[0.0, 0.0, 0.0]])
        for i in range(1, seed_n):
            angle = 2 * np.pi * i / seed_n
            seed_pos = np.vstack([seed_pos, [1.0*np.cos(angle), 1.0*np.sin(angle), 0.0]])
    
    # Relax the seed
    try:
        e_seed, seed_pos = relax(seed_pos, potential, kind)
    except Exception:
        pass
    
    current_pos = seed_pos
    current_n = seed_n
    
    # Sequentially add atoms
    while current_n < n and _time_left() > 30.0:
        # Generate candidate positions for the new atom
        candidates = []
        
        # 1. Try positions near existing atoms (interstitial sites)
        for i in range(current_n):
            for j in range(i+1, current_n):
                # Midpoint between atoms i and j
                mid = (current_pos[i] + current_pos[j]) / 2.0
                # Check if this is a valid position (not too close to any atom)
                dists = np.linalg.norm(current_pos - mid, axis=1)
                if np.min(dists) > 0.4:
                    candidates.append(mid)
        
        # 2. Try positions on a sphere around the cluster
        centroid = np.mean(current_pos, axis=0)
        radius = np.max(np.linalg.norm(current_pos - centroid, axis=1)) + 0.5
        for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
            for phi in np.linspace(0, np.pi, 8, endpoint=False):
                p = centroid + radius * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                dists = np.linalg.norm(current_pos - p, axis=1)
                if np.min(dists) > 0.4:
                    candidates.append(p)
        
        # 3. Random positions
        rng = np.random.default_rng(current_n * 1000 + int(time.time() * 1000) % 10000)
        for _ in range(20):
            p = centroid + rng.uniform(-radius, radius, 3)
            dists = np.linalg.norm(current_pos - p, axis=1)
            if np.min(dists) > 0.4:
                candidates.append(p)
        
        if not candidates:
            # Fallback: just add a random position
            p = centroid + np.array([1.0, 0.0, 0.0])
            candidates = [p]
        
        candidates = np.array(candidates)
        
        # Score each candidate by computing energy of (current_n + 1)-atom cluster
        best_candidate = None
        best_energy = np.inf
        best_new_pos = None
        
        for cand in candidates[:50]:  # Limit to 50 candidates for speed
            new_pos = np.vstack([current_pos, cand])
            if not _is_valid(new_pos):
                continue
            try:
                e, _ = sutton_chen_energy_gradient(new_pos, kind)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_candidate = cand
                    best_new_pos = new_pos
            except Exception:
                continue
        
        if best_new_pos is None:
            # Fallback: use the first valid candidate
            for cand in candidates:
                new_pos = np.vstack([current_pos, cand])
                if _is_valid(new_pos):
                    best_new_pos = new_pos
                    break
            if best_new_pos is None:
                best_new_pos = np.vstack([current_pos, centroid + np.array([1.0, 0.0, 0.0])])
        
        # Relax the new cluster
        try:
            e_relaxed, relaxed_pos = relax(best_new_pos, potential, kind)
            if np.isfinite(e_relaxed) and _is_valid(relaxed_pos):
                current_pos = relaxed_pos
            else:
                current_pos = best_new_pos
        except Exception:
            current_pos = best_new_pos
        
        current_n += 1
    
    return current_pos

def _basin_hopping(positions, kind, potential, max_steps=20, step_size=0.5):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    current_pos = positions.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    for step in range(max_steps):
        if _time_left() < 10.0:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        if not _is_valid(perturbed):
            continue
        
        # Relax
        try:
            e_relaxed, relaxed_pos = relax(perturbed, potential, kind)
            if not np.isfinite(e_relaxed) or not _is_valid(relaxed_pos):
                continue
            
            # Metropolis acceptance
            if e_relaxed < current_energy:
                current_energy = e_relaxed
                current_pos = relaxed_pos
            else:
                # Accept with some probability (simplified: always reject for now)
                pass
        except Exception:
            continue
    
    return current_pos

def construct(instance):
    """
    Construct a low-energy cluster configuration for the given instance.
    """
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: Growth algorithm
    if _time_left() > 60.0:
        try:
            grown_pos = _growth_algorithm(n, kind, potential)
            if _is_valid(grown_pos):
                e_grown, _ = sutton_chen_energy_gradient(grown_pos, kind)
                if np.isfinite(e_grown) and e_grown < best_energy:
                    best_energy = e_grown
                    best_pos = grown_pos
        except Exception:
            pass
    
    # Strategy 2: Basin hopping from lattice starts
    if _time_left() > 30.0:
        starts = []
        pos_fcc = _generate_fcc(n)
        if pos_fcc is not None:
            starts.append(pos_fcc)
        pos_bcc = _generate_bcc(n)
        if pos_bcc is not None:
            starts.append(pos_bcc)
        pos_ico = _generate_icosahedron(n)
        if pos_ico is not None:
            starts.append(pos_ico)
        
        for pos in starts:
            if _time_left() < 15.0:
                break
            try:
                bh_pos = _basin_hopping(pos, kind, potential, max_steps=10)
                if _is_valid(bh_pos):
                    e_bh, _ = sutton_chen_energy_gradient(bh_pos, kind)
                    if np.isfinite(e_bh) and e_bh < best_energy:
                        best_energy = e_bh
                        best_pos = bh_pos
            except Exception:
                continue
    
    # Strategy 3: Direct relaxation of lattice starts (fallback)
    if _time_left() > 10.0:
        starts = []
        pos_fcc = _generate_fcc(n)
        if pos_fcc is not None:
            starts.append(pos_fcc)
        pos_bcc = _generate_bcc(n)
        if pos_bcc is not None:
            starts.append(pos_bcc)
        pos_ico = _generate_icosahedron(n)
        if pos_ico is not None:
            starts.append(pos_ico)
        
        for pos in starts:
            if _time_left() < 5.0:
                break
            try:
                e, relaxed_pos = relax(pos, potential, kind)
                if np.isfinite(e) and _is_valid(relaxed_pos) and e < best_energy:
                    best_energy = e
                    best_pos = relaxed_pos
            except Exception:
                continue
    
    if best_pos is None:
        # Emergency fallback
        return _generate_fcc(n) if _generate_fcc(n) is not None else np.array([[0.0, 0.0, 0.0]])
    
    return best_pos
