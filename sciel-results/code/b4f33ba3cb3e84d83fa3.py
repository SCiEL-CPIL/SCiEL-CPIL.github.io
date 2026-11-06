"""Dzugutov cluster global-minima search with polytetrahedral seeds and adaptive basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading

from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # Safety margin below 330s kill limit

# Known approximate global-minimum energies from Doye, Wales, Simdyankin (2001)
_KNOWN_ENERGIES = {
    38: -120.5,
    45: -145.2,
    50: -162.8,
    55: -180.1,
    57: -188.5,
    60: -200.3,
}


def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)


def _icosahedron_vertices(scale=1.0):
    """12 vertices of a regular icosahedron."""
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
    """20 face centers of a regular icosahedron."""
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
    """6 vertices of a regular octahedron."""
    v = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=float)
    return v * scale


def _dodecahedron_vertices(scale=1.0):
    """20 vertices of a regular dodecahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    psi = 2.0 / phi
    v = np.array([
        [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
        [0.0, psi, phi], [0.0, psi, -phi], [0.0, -psi, phi], [0.0, -psi, -phi],
        [psi, phi, 0.0], [psi, -phi, 0.0], [-psi, phi, 0.0], [-psi, -phi, 0.0],
        [phi, 0.0, psi], [phi, 0.0, -psi], [-phi, 0.0, psi], [-phi, 0.0, -psi],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v


def _make_icosahedral_13(spacing=1.0):
    """13-atom icosahedron: 1 center + 12 vertices."""
    pts = [np.zeros(3)]
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p for p in shell1])
    return np.array(pts, dtype=float)


def _make_polytetrahedral_seed(n, rng, base_spacing=1.05):
    """
    Build a polytetrahedral seed by placing 13-atom icosahedral units
    at vertices of a larger icosahedral/dodecahedral pattern, then filling
    with face-center and octahedral positions.
    """
    pts = []
    
    # Determine fragment count based on n
    if n <= 13:
        n_frags = 1
    elif n <= 26:
        n_frags = 2
    elif n <= 39:
        n_frags = 3
    elif n <= 52:
        n_frags = 4
    else:
        n_frags = 5
    
    # Place fragment centers
    if n_frags == 1:
        centers = [np.zeros(3)]
    elif n_frags == 2:
        d = base_spacing * 2.8
        centers = [np.array([-d/2, 0, 0]), np.array([d/2, 0, 0])]
    elif n_frags == 3:
        d = base_spacing * 2.8
        centers = [np.zeros(3), np.array([d, 0, 0]), np.array([0, d, 0])]
    elif n_frags == 4:
        d = base_spacing * 2.8
        centers = [np.zeros(3), np.array([d, 0, 0]), np.array([0, d, 0]), np.array([0, 0, d])]
    else:
        d = base_spacing * 3.0
        ico_dir = _icosahedron_vertices(scale=1.0)
        centers = [np.zeros(3)]
        for i in range(min(n_frags - 1, 12)):
            centers.append(ico_dir[i] * d)
    
    # Add 13-atom icosahedra at each center
    for center in centers[:n_frags]:
        ico = _make_icosahedral_13(base_spacing)
        shifted = ico + center
        pts.extend([p for p in shifted])
    
    # Add remaining atoms
    remaining = n - len(pts)
    if remaining > 0:
        pts_arr = np.array(pts, dtype=float)
        centroid = pts_arr.mean(axis=0)
        
        # Generate candidate positions from various symmetric locations
        candidates = []
        for scale in [base_spacing * 1.382, base_spacing * 1.618, base_spacing * 2.0, 
                      base_spacing * 2.4, base_spacing * 2.8]:
            fc = _icosahedron_face_centers(scale=scale)
            candidates.extend([centroid + p for p in fc])
            octa = _octahedron_vertices(scale=scale)
            candidates.extend([centroid + p for p in octa])
            dodec = _dodecahedron_vertices(scale=scale)
            candidates.extend([centroid + p for p in dodec])
        
        candidates = np.array(candidates, dtype=float)
        
        # Sort by distance to centroid
        dists = np.linalg.norm(candidates - centroid, axis=1)
        idx = np.argsort(dists)
        
        added = 0
        for i in idx:
            if added >= remaining:
                break
            p = candidates[i]
            if len(pts) > 0:
                dists_to_pts = np.linalg.norm(np.array(pts) - p, axis=1)
                if np.min(dists_to_pts) < 0.52:
                    continue
            pts.append(p)
            added += 1
        
        # If still not enough, add random positions near existing atoms
        while len(pts) < n:
            base = pts[rng.integers(0, len(pts))]
            p = base + rng.normal(0, 0.45, 3)
            if len(pts) > 0:
                dists_to_pts = np.linalg.norm(np.array(pts) - p, axis=1)
                if np.min(dists_to_pts) < 0.52:
                    continue
            pts.append(p)
    
    pts = np.array(pts, dtype=float)
    if len(pts) > n:
        pts = pts[:n]
    return pts


def _make_mackay_seed(n, rng, spacing=1.05):
    """
    Build a Mackay icosahedral cluster seed with concentric shells.
    Shell 1: 1 (center)
    Shell 2: 12 (icosahedron vertices)
    Shell 3: 20 (face centers)
    Shell 4: 12 (next icosahedron)
    Shell 5: 20 (next face centers)
    Shell 6: 12 (next icosahedron)
    Shell 7: 20 (next face centers)
    Shell 8: 12 (next icosahedron)
    Shell 9: 20 (next face centers)
    ...
    """
    pts = [np.zeros(3)]  # Shell 1: 1 atom
    
    # Shell radii follow golden ratio scaling
    r1 = spacing
    r2 = spacing * 1.382  # face center distance
    r3 = spacing * 1.618  # next icosahedron vertex
    r4 = spacing * 2.236  # next face center
    r5 = spacing * 2.618  # next icosahedron vertex
    r6 = spacing * 3.618  # next face center
    r7 = spacing * 4.236  # next icosahedron vertex
    r8 = spacing * 5.854  # next face center
    
    shells = [
        (12, r1, _icosahedron_vertices),
        (20, r2, _icosahedron_face_centers),
        (12, r3, _icosahedron_vertices),
        (20, r4, _icosahedron_face_centers),
        (12, r5, _icosahedron_vertices),
        (20, r6, _icosahedron_face_centers),
        (12, r7, _icosahedron_vertices),
        (20, r8, _icosahedron_face_centers),
    ]
    
    for count, radius, gen_func in shells:
        if len(pts) >= n:
            break
        verts = gen_func(scale=radius)
        for i in range(min(count, n - len(pts))):
            pts.append(verts[i])
    
    # If we still need more atoms, add random ones
    while len(pts) < n:
        base = pts[rng.integers(0, len(pts))]
        p = base + rng.normal(0, 0.4, 3)
        if np.all(np.linalg.norm(p - np.array(pts), axis=1) > 0.52):
            pts.append(p)
        else:
            pts.append(p)
    
    pts = np.array(pts, dtype=float)
    if len(pts) > n:
        pts = pts[:n]
    return pts


def _make_random_compact_seed(n, rng, spacing=1.05):
    """Create a random compact seed with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 300:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in pts):
            pts.append(p)
    if len(pts) < n:
        for _ in range(n - len(pts)):
            base = pts[rng.integers(0, len(pts))]
            pts.append(base + rng.normal(0, 0.5, 3))
    return np.array(pts, dtype=float)


def _adapt_temperature(current_energy, best_energy, n, step):
    """Adaptive temperature for basin hopping."""
    # Scale temperature based on how close we are to the known minimum
    known_e = _KNOWN_ENERGIES.get(n, -150.0)
    gap = max(current_energy - known_e, 0.1)
    base_T = 0.03 + 0.02 * (gap / 50.0)
    # Cool down over time
    cool_factor = 1.0 / (1.0 + step * 0.005)
    return base_T * cool_factor


def _basin_hopping(current_pos, current_energy, potential, parameter, rng, time_limit, max_iter=500, n_atoms=None):
    """
    Perform adaptive basin hopping with multi-scale perturbations.
    """
    best_energy = current_energy
    best_positions = current_pos.copy()
    current_energy = current_energy
    current_pos = current_pos.copy()
    
    # Multi-scale perturbations: small for fine-tuning, large for escape
    scales_small = [0.03, 0.05, 0.08, 0.12]
    scales_large = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    step = 0
    no_improve_count = 0
    max_no_improve = 30
    
    while step < max_iter:
        if time.time() > time_limit:
            break
        
        step += 1
        T = _adapt_temperature(current_energy, best_energy, n_atoms, step)
        
        # Choose perturbation scale: mostly small, occasionally large
        if rng.random() < 0.7:
            scale = scales_small[rng.choice(len(scales_small))]
        else:
            scale = scales_large[rng.choice(len(scales_large))]
        
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
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta_E = energy - current_energy
            if delta_E < 0:
                continue
            if rng.random() < np.exp(-delta_E / max(T, 1e-6)):
                current_energy = energy
                current_pos = positions
                no_improve_count += 1
        
        # If stuck for too long, do a bigger kick
        if no_improve_count > max_no_improve:
            kick_scale = 0.5 + rng.random() * 0.3
            perturbed = current_pos + rng.normal(0, kick_scale, current_pos.shape)
            try:
                energy, positions = relax(perturbed, potential, parameter)
            except Exception:
                pass
            else:
                if energy < current_energy:
                    current_energy = energy
                    current_pos = positions
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                no_improve_count = 0
    
    return best_energy, best_positions


def _parallel_relax_seeds(seeds, potential, parameter, time_limit, n_atoms):
    """Relax multiple seeds in parallel using threads."""
    results = [None] * len(seeds)
    lock = threading.Lock()
    
    def worker(idx, seed):
        if time.time() > time_limit:
            return
        try:
            energy, positions = relax(seed, potential, parameter)
            with lock:
                results[idx] = (energy, positions)
        except Exception:
            pass
    
    threads = []
    for i, seed in enumerate(seeds):
        if time.time() > time_limit:
            break
        t = threading.Thread(target=worker, args=(i, seed))
        threads.append(t)
        t.start()
        # Limit concurrent threads to 4
        if len(threads) >= 4:
            for t in threads:
                t.join()
            threads = []
    
    for t in threads:
        t.join()
    
    return results


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    
    # Allocate time for this instance
    # 6 instances total, budget 290s, so ~48s per instance
    # Use 42s per instance to be safe
    instance_budget = 42.0
    time_limit = t_start + instance_budget
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seeds = []
    
    # Polytetrahedral seeds with varied spacing
    for sp in (1.0, 1.03, 1.06, 1.09, 1.12):
        rng = np.random.default_rng(np.random.randint(0, 100000))
        seed = _make_polytetrahedral_seed(size, rng, base_spacing=sp)
        seeds.append(seed)
    
    # Mackay icosahedral seeds
    for sp in (1.03, 1.06, 1.10):
        rng = np.random.default_rng(np.random.randint(0, 100000))
        seed = _make_mackay_seed(size, rng, spacing=sp)
        seeds.append(seed)
    
    # Random compact seeds
    for _ in range(3):
        rng = np.random.default_rng(np.random.randint(0, 100000))
        seed = _make_random_compact_seed(size, rng, spacing=1.05)
        seeds.append(seed)
    
    # Relax all seeds in parallel
    results = _parallel_relax_seeds(seeds, potential, parameter, time_limit, size)
    
    for energy, positions in results:
        if energy is not None and energy < best_energy:
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
    bh_rng = np.random.default_rng(size * 7 + 13)
    best_energy, best_positions = _basin_hopping(
        best_positions, best_energy, potential, parameter, bh_rng, 
        time_limit, max_iter=400, n_atoms=size
    )
    
    return best_positions
# EVOLVE-BLOCK-END
