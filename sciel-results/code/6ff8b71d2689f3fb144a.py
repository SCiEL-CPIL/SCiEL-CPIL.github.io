"""Dzugutov cluster global-minima search with fragment-merger initialization and multi-scale basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading

from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin below 330s kill limit


def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)


def _icosahedron_vertices(scale=1.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v


def _icosahedron_face_centers(scale=1.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    fc = np.array([
        [1.0, 1.0, phi], [1.0, -1.0, phi], [1.0, 1.0, -phi], [1.0, -1.0, -phi],
        [-1.0, 1.0, phi], [-1.0, -1.0, phi], [-1.0, 1.0, -phi], [-1.0, -1.0, -phi],
        [1.0, phi, 1.0], [-1.0, phi, 1.0], [1.0, phi, -1.0], [-1.0, phi, -1.0],
        [1.0, -phi, 1.0], [-1.0, -phi, 1.0], [1.0, -phi, -1.0], [-1.0, -phi, -1.0],
        [phi, 1.0, 1.0], [-phi, 1.0, 1.0], [phi, -1.0, 1.0], [-phi, -1.0, 1.0],
    ], dtype=float)
    norms = np.linalg.norm(fc, axis=1, keepdims=True)
    fc = fc / norms * scale
    return fc


def _octahedron_vertices(scale=1.0):
    v = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=float)
    return v * scale


def _make_icosahedral_13(spacing=1.0):
    """13-atom icosahedron: 1 center + 12 vertices."""
    pts = [np.zeros(3)]
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p for p in shell1])
    return np.array(pts, dtype=float)


def _make_icosahedral_32(spacing=1.0):
    """32-atom icosahedron: 13 + 20 face centers + 6 octahedral directions (approx)."""
    pts = list(_make_icosahedral_13(spacing))
    # Add 20 face centers at radius ~1.382*spacing
    fc = _icosahedron_face_centers(scale=spacing * 1.382)
    pts.extend([p for p in fc])
    # Add 6 octahedral directions at radius ~1.618*spacing
    octa = _octahedron_vertices(scale=spacing * 1.618)
    pts.extend([p for p in octa])
    # We now have 13 + 20 + 6 = 39, take first 32
    pts = np.array(pts, dtype=float)
    if len(pts) > 32:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:32]
        pts = pts[idx]
    return pts


def _make_polytetrahedral_seed(n, rng, base_spacing=1.05):
    """
    Build a polytetrahedral seed by merging small icosahedral fragments.
    Strategy: place 13-atom icosahedra at centers of a larger icosahedral pattern,
    then add remaining atoms at face-center positions.
    """
    pts = []
    
    # Determine how many 13-atom fragments to use
    n_fragments = max(1, n // 13)
    remaining = n - n_fragments * 13
    
    # Place fragment centers on an icosahedral pattern
    if n_fragments == 1:
        centers = [np.zeros(3)]
    elif n_fragments == 2:
        d = base_spacing * 2.5
        centers = [np.array([-d/2, 0, 0]), np.array([d/2, 0, 0])]
    elif n_fragments == 3:
        d = base_spacing * 2.5
        centers = [np.array([0, 0, 0]), np.array([d, 0, 0]), np.array([0, d, 0])]
    elif n_fragments == 4:
        d = base_spacing * 2.5
        centers = [np.array([0, 0, 0]), np.array([d, 0, 0]), np.array([0, d, 0]), np.array([0, 0, d])]
    else:
        # For more fragments, use icosahedral directions
        ico_dir = _icosahedron_vertices(scale=1.0)
        d = base_spacing * 3.0
        centers = [np.zeros(3)]
        for i in range(min(n_fragments - 1, 12)):
            centers.append(ico_dir[i] * d)
    
    # For each center, add a 13-atom icosahedron
    for center in centers[:n_fragments]:
        ico = _make_icosahedral_13(base_spacing)
        shifted = ico + center
        pts.extend([p for p in shifted])
    
    # Add remaining atoms at face-center positions of the overall structure
    if remaining > 0:
        # Compute centroid of all points so far
        pts_arr = np.array(pts, dtype=float)
        centroid = pts_arr.mean(axis=0)
        
        # Generate candidate positions from icosahedral face centers and octahedral directions
        candidates = []
        for scale in [base_spacing * 1.382, base_spacing * 1.618, base_spacing * 2.0, base_spacing * 2.5]:
            fc = _icosahedron_face_centers(scale=scale)
            candidates.extend([centroid + p for p in fc])
            octa = _octahedron_vertices(scale=scale)
            candidates.extend([centroid + p for p in octa])
        
        candidates = np.array(candidates, dtype=float)
        
        # Sort candidates by distance to centroid (prefer closer ones)
        dists = np.linalg.norm(candidates - centroid, axis=1)
        idx = np.argsort(dists)
        
        added = 0
        for i in idx:
            if added >= remaining:
                break
            p = candidates[i]
            # Check minimum distance to existing points
            if len(pts) > 0:
                dists_to_pts = np.linalg.norm(np.array(pts) - p, axis=1)
                if np.min(dists_to_pts) < 0.55:
                    continue
            pts.append(p)
            added += 1
        
        # If still not enough, add random perturbations
        while len(pts) < n:
            base = pts[rng.integers(0, len(pts))]
            p = base + rng.normal(0, 0.4, 3)
            if len(pts) > 0:
                dists_to_pts = np.linalg.norm(np.array(pts) - p, axis=1)
                if np.min(dists_to_pts) < 0.55:
                    continue
            pts.append(p)
    
    pts = np.array(pts, dtype=float)
    if len(pts) > n:
        pts = pts[:n]
    return pts


def _make_icosahedral_cluster_seed(n, rng, spacing=1.05):
    """
    Build an icosahedral cluster seed with multiple shells.
    """
    pts = [np.zeros(3)]
    
    # Shell 1: 12 icosahedral vertices
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p for p in shell1])
    
    # Shell 2: 20 face centers
    fc = _icosahedron_face_centers(scale=spacing * 1.382)
    pts.extend([p for p in fc])
    
    # Shell 3: 6 octahedral directions
    octa = _octahedron_vertices(scale=spacing * 1.618)
    pts.extend([p for p in octa])
    
    # Shell 4: 12 icosahedral at larger radius
    shell4 = _icosahedron_vertices(scale=spacing * 2.0)
    pts.extend([p for p in shell4])
    
    # Shell 5: 20 face centers at larger radius
    fc5 = _icosahedron_face_centers(scale=spacing * 2.618)
    pts.extend([p for p in fc5])
    
    # Shell 6: 6 octahedral at larger radius
    octa6 = _octahedron_vertices(scale=spacing * 2.618)
    pts.extend([p for p in octa6])
    
    pts = np.array(pts, dtype=float)
    
    if len(pts) >= n:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n]
        pts = pts[idx]
    else:
        while len(pts) < n:
            base = pts[rng.integers(0, len(pts))]
            p = base + rng.normal(0, 0.35, 3)
            if np.all(np.linalg.norm(p - pts, axis=1) > 0.55):
                pts = np.vstack([pts, p])
            else:
                p = base + rng.normal(0, 0.7, 3)
                if np.all(np.linalg.norm(p - pts, axis=1) > 0.55):
                    pts = np.vstack([pts, p])
                else:
                    pts = np.vstack([pts, p])
    return pts


def _make_random_compact_seed(n, rng, spacing=1.05):
    """Create a random compact seed with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 200:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in pts):
            pts.append(p)
    if len(pts) < n:
        for _ in range(n - len(pts)):
            base = pts[rng.integers(0, len(pts))]
            pts.append(base + rng.normal(0, 0.5, 3))
    return np.array(pts, dtype=float)


def _basin_hopping(current_pos, current_energy, potential, parameter, rng, time_limit, max_iter=200):
    """
    Perform basin hopping with multi-scale perturbations.
    """
    best_energy = current_energy
    best_positions = current_pos.copy()
    current_energy = current_energy
    current_pos = current_pos.copy()
    
    scales = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    T = 0.05
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
        
        scale = scales[rng.choice(len(scales))]
        perturbed = current_pos + rng.normal(0, scale, current_pos.shape)
        
        try:
            energy, positions = relax(perturbed, potential, parameter)
        except Exception:
            continue
        
        # Metropolis acceptance
        if energy < current_energy:
            current_energy = energy
            current_pos = positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        else:
            delta_E = energy - current_energy
            if delta_E < 0:
                continue
            if rng.random() < np.exp(-delta_E / T):
                current_energy = energy
                current_pos = positions
    
    return best_energy, best_positions


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    
    # Allocate time for this instance.
    # We have 6 instances. Total budget 295s.
    # Let's assume equal distribution but with a safety margin.
    # 295 / 6 = ~49s per instance.
    # Let's use 45s per instance to be safe.
    instance_budget = 45.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds using fragment-merger and icosahedral approaches
    seeds = []
    
    # Fragment-merger seeds with varied spacing
    for sp in (1.02, 1.05, 1.08, 1.10, 1.12):
        rng = np.random.default_rng(np.random.randint(0, 10000))
        seed = _make_polytetrahedral_seed(size, rng, base_spacing=sp)
        seeds.append(seed)
    
    # Icosahedral cluster seeds with varied spacing
    for sp in (1.05, 1.10, 1.15):
        rng = np.random.default_rng(np.random.randint(0, 10000))
        seed = _make_icosahedral_cluster_seed(size, rng, spacing=sp)
        seeds.append(seed)
    
    # Random compact seeds
    for _ in range(4):
        rng = np.random.default_rng(np.random.randint(0, 10000))
        seed = _make_random_compact_seed(size, rng, spacing=1.05)
        seeds.append(seed)
    
    # Relax all seeds
    for seed in seeds:
        if (time.time() - t_start) > instance_budget:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
        except Exception:
            continue
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    if best_positions is None:
        # Fallback
        seed = _make_random_compact_seed(size, np.random.default_rng(1), spacing=1.05)
        try:
            _, best_positions = relax(seed, potential, parameter)
        except Exception:
            best_positions = seed
        return best_positions
    
    # Basin Hopping from the best solution
    bh_rng = np.random.default_rng(size + 100)
    
    # Use remaining time for basin hopping
    time_limit = t_start + instance_budget
    best_energy, best_positions = _basin_hopping(
        best_positions, best_energy, potential, parameter, bh_rng, time_limit, max_iter=300
    )
    
    return best_positions
# EVOLVE-BLOCK-END
