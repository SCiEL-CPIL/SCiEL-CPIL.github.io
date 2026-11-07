import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit circumradius)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _icosahedral_cluster(n):
    """Generate an icosahedral cluster with n atoms using Mackay shells."""
    verts = _get_icosahedron_vertices()
    # Shell radii for icosahedral clusters (relative to nearest-neighbor distance ~1.0)
    # Shell 0: center (1 atom)
    # Shell 1: 12 vertices at r=1
    # Shell 2: 30 points at r=sqrt(3) ≈ 1.732 (edge midpoints, scaled)
    # Shell 3: 20 points at r=2 (face centers, scaled)
    # Shell 4: 60 points at r=sqrt(5) ≈ 2.236
    # Shell 5: 120 points at r=sqrt(7) ≈ 2.646
    
    pts = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    shell1 = verts * 1.0
    pts.extend(shell1.tolist())
    
    # Shell 2: 30 points - edge midpoints of icosahedron, scaled to proper radius
    # Find edges (pairs of vertices at distance ~1.0515 for unit circumradius)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:  # Adjacent vertices
                edges.append((i, j))
    
    shell2 = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid = mid / mid_norm * np.sqrt(3.0)
            shell2.append(mid)
    pts.extend(shell2)
    
    # Shell 3: 20 points - face centers, scaled to r=2
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    faces.append((i, j, k))
    
    shell3 = []
    for i, j, k in faces:
        centroid = (verts[i] + verts[j] + verts[k]) / 3.0
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-10:
            centroid = centroid / centroid_norm * 2.0
            shell3.append(centroid)
    pts.extend(shell3)
    
    # Shell 4: 60 points at r=sqrt(5)
    # These are positions like (0, ±phi, ±1) normalized and scaled, etc.
    # Use all permutations of (0, 1, phi) with signs, scaled to r=sqrt(5)
    shell4_base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0],
        [1, phi, 1], [-1, phi, 1], [1, -phi, 1], [-1, -phi, 1],
        [1, phi, -1], [-1, phi, -1], [1, -phi, -1], [-1, -phi, -1],
        [phi, 1, 1], [-phi, 1, 1], [phi, -1, 1], [-phi, -1, 1],
        [phi, 1, -1], [-phi, 1, -1], [phi, -1, -1], [-phi, -1, -1],
    ], dtype=float)
    # Remove duplicates and scale
    shell4 = []
    seen = set()
    for v in shell4_base:
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        v_scaled = v / v_norm * np.sqrt(5.0)
        key = tuple(np.round(v_scaled, 6))
        if key not in seen:
            seen.add(key)
            shell4.append(v_scaled)
    pts.extend(shell4)
    
    # Shell 5: 120 points at r=sqrt(7)
    # Generate from all (±1, ±1, ±phi) type combinations
    shell5_base = []
    for a in [1, -1]:
        for b in [1, -1]:
            for c in [phi, -phi]:
                for perm in [(a, b, c), (b, c, a), (c, a, b)]:
                    shell5_base.append(perm)
    shell5 = []
    seen = set()
    for v in shell5_base:
        v = np.array(v, dtype=float)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        v_scaled = v / v_norm * np.sqrt(7.0)
        key = tuple(np.round(v_scaled, 6))
        if key not in seen:
            seen.add(key)
            shell5.append(v_scaled)
    pts.extend(shell5)
    
    pts = np.array(pts)
    
    # If we have more than n, take the closest n to origin
    if len(pts) > n:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)
        pts = pts[idx[:n]]
    elif len(pts) < n:
        # Fill remaining with additional points on a sphere
        rng = np.random.default_rng(42)
        extra = []
        for _ in range(n - len(pts)):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * (np.sqrt(7.0) + rng.uniform(0, 0.5))
            extra.append(v)
        pts = np.vstack([pts, np.array(extra)])
    
    return pts[:n]

def _fcc_cluster(n):
    """Generate n points from an FCC lattice closest to origin."""
    max_coord = int(np.ceil((n ** (1/3.0)) * 2.5)) + 5
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: positions (i,j,k)/2 where i+j+k even, plus basis
                base = np.array([i, j, k]) * 0.5
                offsets = [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.0, 0.5, 0.5]),
                ]
                for off in offsets:
                    pts.append(base + off)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _bcc_cluster(n):
    """Generate n points from a BCC lattice closest to origin."""
    max_coord = int(np.ceil((n ** (1/3.0)) * 2.5)) + 5
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                pts.append([0.5*i, 0.5*j, 0.5*k])
                if (i + j + k) % 2 == 0:
                    pts.append([0.5*i + 0.5, 0.5*j + 0.5, 0.5*k + 0.5])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _hcp_cluster(n):
    """Generate n points from an HCP lattice closest to origin."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    max_coord = int(np.ceil((n ** (1/3.0)) * 2.5)) + 5
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i + 0.5 * j
            y = np.sqrt(3.0)/2.0 * j
            pts.append([x, y, 0.0])
            x_shift = x + a/3.0
            y_shift = y + np.sqrt(3.0)*a/6.0
            pts.append([x_shift, y_shift, c/2.0])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _random_compact(n, seed=42):
    """Generate n random points in a compact spherical region with minimum separation."""
    rng = np.random.default_rng(seed)
    # Estimate radius for n points with typical interatomic distance ~1
    radius = (n / (4.0/3.0 * np.pi)) ** (1/3.0) * 1.2
    points = []
    min_dist = 0.6
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: place remaining points randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _validate(pos, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                return False
    return True

def _fix_close_atoms(pos, min_dist=0.35):
    """Separate any atoms closer than min_dist."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(10):  # Multiple passes
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    if d < 1e-10:
                        pos[j] += np.array([0.01, 0.01, 0.01])
                    else:
                        direction = (pos[j] - pos[i]) / d
                        pos[j] = pos[i] + direction * min_dist
                    changed = True
        if not changed:
            break
    return pos

def _basin_hopping(pos, kind, n_max_iter=30, kick_size=0.3, time_limit=30.0):
    """
    Basin hopping: local relax + random kick + accept/reject.
    Returns (best_energy, best_positions).
    """
    start_time = time.time()
    best_energy = np.inf
    best_pos = pos.copy()
    current_energy = np.inf
    current_pos = pos.copy()
    rng = np.random.default_rng(12345)
    
    # Initial relaxation
    try:
        current_energy, current_pos = relax(pos, "sutton_chen", kind)
        best_energy = current_energy
        best_pos = current_pos.copy()
    except Exception:
        return np.inf, pos
    
    for it in range(n_max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random kick
        kick = rng.normal(0, kick_size, size=current_pos.shape)
        proposed = current_pos + kick
        proposed = _fix_close_atoms(proposed)
        
        if not _validate(proposed):
            continue
        
        # Relax proposed
        try:
            new_energy, new_pos = relax(proposed, "sutton_chen", kind)
        except Exception:
            continue
        
        # Accept if lower energy
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos.copy()
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        # Simple acceptance: always accept if lower, otherwise with some probability
        elif new_energy < current_energy + 2.0:
            # Accept with probability based on energy difference
            prob = np.exp(-(new_energy - current_energy) / 1.0)
            if rng.random() < prob:
                current_energy = new_energy
                current_pos = new_pos.copy()
        
        # Adapt kick size
        if it % 10 == 0 and it > 0:
            kick_size *= 0.95  # Slowly decrease kick size
    
    return best_energy, best_pos

def _relax_and_record(pos, kind, results, lock):
    """Relax a single candidate and record result. Thread-safe."""
    try:
        energy, relaxed = relax(pos, "sutton_chen", kind)
        with lock:
            results.append((energy, relaxed))
    except Exception:
        pass

def construct(instance):
    """
    Construct a low-energy cluster configuration using basin hopping
    with diverse structural initializations.
    """
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = 310.0  # Safety margin within 330s limit
    
    # Generate diverse initial candidates
    candidates = []
    
    # Icosahedral cluster (most physically relevant for TM clusters)
    try:
        candidates.append(_icosahedral_cluster(n))
    except Exception:
        pass
    
    # FCC, BCC, HCP
    try:
        candidates.append(_fcc_cluster(n))
    except Exception:
        pass
    try:
        candidates.append(_bcc_cluster(n))
    except Exception:
        pass
    try:
        candidates.append(_hcp_cluster(n))
    except Exception:
        pass
    
    # Random compact starts
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        try:
            candidates.append(_random_compact(n, seed=seed))
        except Exception:
            pass
    
    # Validate and fix all candidates
    valid_candidates = []
    for pos in candidates:
        try:
            pos = np.array(pos, dtype=float)
            pos = _fix_close_atoms(pos)
            if _validate(pos):
                valid_candidates.append(pos)
        except Exception:
            continue
    
    if not valid_candidates:
        rng = np.random.default_rng(0)
        pos = rng.uniform(-2, 2, (n, 3))
        pos = _fix_close_atoms(pos)
        valid_candidates.append(pos)
    
    # Phase 1: Quick relax all candidates to get initial energies
    best_energy = np.inf
    best_positions = None
    relaxed_results = []
    
    import threading
    lock = threading.Lock()
    
    # Use ThreadPoolExecutor for parallel relaxation
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for pos in valid_candidates:
            if time.time() - start_time > deadline - 30:
                break
            futures.append(executor.submit(_relax_and_record, pos, kind, relaxed_results, lock))
        for f in as_completed(futures):
            pass
    
    # Find best from relaxed results
    for energy, pos in relaxed_results:
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    if best_positions is None:
        best_positions = valid_candidates[0]
        best_energy = np.inf
    
    # Phase 2: Basin hopping from the best few starting points
    # Sort relaxed results by energy and take top few
    relaxed_results.sort(key=lambda x: x[0])
    top_starts = [pos for _, pos in relaxed_results[:5]] if relaxed_results else [best_positions]
    
    bh_best_energy = best_energy
    bh_best_pos = best_positions.copy()
    
    # Run basin hopping from each top start
    for start_pos in top_starts:
        if time.time() - start_time > deadline - 10:
            break
        
        # Allocate remaining time for this basin hopping run
        remaining = deadline - 10 - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Use more iterations if we have more time
        n_iter = max(20, int(remaining / 2.0))
        n_iter = min(n_iter, 80)  # Cap iterations
        
        try:
            bh_energy, bh_pos = _basin_hopping(
                start_pos, kind, 
                n_max_iter=n_iter, 
                kick_size=0.25,
                time_limit=remaining
            )
            if bh_energy < bh_best_energy:
                bh_best_energy = bh_energy
                bh_best_pos = bh_pos
        except Exception:
            continue
    
    # Final result
    if bh_best_pos is not None and _validate(bh_best_pos):
        return bh_best_pos
    elif _validate(best_positions):
        return best_positions
    else:
        # Last resort
        pos = valid_candidates[0]
        pos = _fix_close_atoms(pos)
        return pos
