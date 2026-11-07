"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return normalized icosahedron vertices (12 points)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _icosahedron_edges():
    """Return edge midpoint directions (30 edges, normalized)."""
    verts = _icosahedron_vertices()
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    edge_pairs = [(dists[k][1], dists[k][2]) for k in range(30)]
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    return np.array(edge_dirs)


def _icosahedron_face_centers():
    """Return 20 face center directions (normalized)."""
    verts = _icosahedron_vertices()
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8],
    ]
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)


def _build_icosahedral_cluster(n, seed=0, radii_scale=1.0, include_center=True, jitter=0.0):
    """Build a cluster with icosahedral core + surface atoms at high-symmetry sites."""
    rng = np.random.default_rng(seed)
    positions = []
    
    # Core icosahedron: 12 atoms
    core_radius = 0.72 * radii_scale
    core = _icosahedron_vertices() * core_radius
    if jitter > 0:
        core += rng.normal(0, jitter, core.shape)
    positions.extend(core)
    
    # Add center atom if n >= 13
    if n >= 13 and include_center:
        positions.append([0.0, 0.0, 0.0])
    
    n_remaining = n - len(positions)
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: 30 edge-midpoint positions
    edge_dirs = _icosahedron_edges()
    shell2_radius = 1.05 * radii_scale
    shell2 = edge_dirs * shell2_radius
    if jitter > 0:
        shell2 += rng.normal(0, jitter, shell2.shape)
        
    n_add2 = min(n_remaining, 30)
    positions.extend(shell2[:n_add2])
    n_remaining -= n_add2
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Third shell: 20 face centers
    face_dirs = _icosahedron_face_centers()
    shell3_radius = 1.35 * radii_scale
    shell3 = face_dirs * shell3_radius
    if jitter > 0:
        shell3 += rng.normal(0, jitter, shell3.shape)

    n_add3 = min(n_remaining, 20)
    positions.extend(shell3[:n_add3])
    n_remaining -= n_add3
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Fourth shell: Fibonacci sphere for remaining atoms
    if n_remaining > 0:
        indices = np.arange(n_remaining)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_remaining)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        shell4_radius = 1.55 * radii_scale
        shell4 = np.column_stack([
            shell4_radius * np.sin(phi) * np.cos(theta),
            shell4_radius * np.sin(phi) * np.sin(theta),
            shell4_radius * np.cos(phi)
        ])
        if jitter > 0:
            shell4 += rng.normal(0, jitter, shell4.shape)
        positions.extend(shell4)
    
    return np.array(positions[:n])


def _build_fcc_fragment(n, seed=0):
    """Build FCC lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    positions -= np.mean(positions, axis=0)
    return positions


def _build_hcp_fragment(n, seed=0):
    """Build HCP lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    c = a * np.sqrt(8.0/3.0)
    layer_spacing = c / 2.0
    
    positions = []
    n_layers = int(np.ceil(np.sqrt(n / 3.0))) + 2
    
    for layer in range(n_layers):
        z = layer * layer_spacing
        offset = np.array([0, 0, 0]) if layer % 2 == 0 else np.array([a/2, a/(2*np.sqrt(3)), 0])
        
        max_r = int(np.ceil(np.sqrt(n / (3 * np.pi / (a**2 * 0.866)))))
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = a * (i + j / 2) + offset[0]
                y = a * np.sqrt(3) / 2 * j + offset[1]
                p = np.array([x, y, z])
                if len(positions) < n:
                    positions.append(p)
                else:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _build_random_cluster(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def _sa_basin_hopping(positions, potential, kind, rng=None, deadline=None, max_iter=40, 
                      t_start=0.5, t_end=0.05, initial_pert=0.4, min_pert=0.1):
    """
    Simulated Annealing Basin Hopping.
    Uses a cooling schedule to accept worse moves early on to escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initial relaxation
    try:
        energy, pos = relax(positions, potential, kind)
    except Exception:
        return np.inf, positions
    
    if not _validate(pos):
        return np.inf, positions
    
    best_energy = energy
    best_pos = pos.copy()
    current_energy = energy
    current_pos = pos.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Cooling schedule: exponential decay
        alpha = t_end / t_start
        T = t_start * (alpha ** (i / max_iter))
        
        # Perturbation scale also decays
        pert_scale = initial_pert * (min_pert / initial_pert) ** (i / max_iter)
        
        noise = rng.normal(0, pert_scale, current_pos.shape)
        perturbed = current_pos + noise
        
        if not _validate(perturbed):
            # Retry with smaller noise if invalid
            noise = rng.normal(0, pert_scale * 0.5, current_pos.shape)
            perturbed = current_pos + noise
            if not _validate(perturbed):
                continue
        
        try:
            e2, pos2 = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if not _validate(pos2):
            continue
            
        delta_e = e2 - current_energy
        
        # Metropolis acceptance criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-9)):
            current_energy = e2
            current_pos = pos2
            
            if e2 < best_energy:
                best_energy = e2
                best_pos = pos2.copy()
                
    return best_energy, best_pos


def construct(instance):
    """Structure prediction with parallel SA Basin Hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for safety (330s total limit)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Strategy 1: Icosahedral variants with different scales and jitters
    radii_scales = [0.9, 1.0, 1.1, 1.2]
    for rs in radii_scales:
        for seed in range(2):
            jitter = 0.05 if seed == 1 else 0.0
            pos = _build_icosahedral_cluster(size, seed=seed, radii_scale=rs, include_center=True, jitter=jitter)
            if _validate(pos):
                initial_configs.append(pos)
    
    # Strategy 2: FCC fragments
    for seed in range(2):
        pos = _build_fcc_fragment(size, seed=seed)
        if _validate(pos):
            initial_configs.append(pos)
            
    # Strategy 3: HCP fragments
    for seed in range(2):
        pos = _build_hcp_fragment(size, seed=seed)
        if _validate(pos):
            initial_configs.append(pos)
            
    # Strategy 4: Random compact clusters
    for seed in range(3):
        pos = _build_random_cluster(size, seed=seed)
        if _validate(pos):
            initial_configs.append(pos)
            
    # If we have too many, subsample to keep time under control
    # With 4 threads, we can run 4 at a time. 
    # Let's aim for ~12-16 total configs to ensure coverage without timeout.
    max_configs = 16
    if len(initial_configs) > max_configs:
        # Prioritize icosahedral and FCC
        ico = [c for c in initial_configs if np.allclose(np.linalg.norm(c, axis=1).mean(), 1.0, atol=0.5)] # Crude filter
        # Just take the first 16 for simplicity and determinism
        initial_configs = initial_configs[:max_configs]
        
    best_energy = np.inf
    best_positions = None
    
    # Use ThreadPoolExecutor for parallel optimization
    # max_workers=4 as per constraint
    
    def worker(config):
        seed = np.random.default_rng().integers(0, 10000)
        rng = np.random.default_rng(seed)
        try:
            energy, pos = _sa_basin_hopping(config, potential, kind, rng=rng, deadline=deadline)
            if _validate(pos):
                return energy, pos
            return np.inf, None
        except Exception:
            return np.inf, None
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, config): config for config in initial_configs}
        
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline:
                break
            try:
                energy, pos = future.result()
                if pos is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
                
    # If no valid result was found, fallback to a single synchronous relaxation
    if best_positions is None:
        positions = _build_icosahedral_cluster(size, seed=0, include_center=True)
        if not _validate(positions):
            positions = _build_random_cluster(size, seed=42)
        try:
            energy, best_positions = relax(positions, potential, kind)
        except Exception:
            best_positions = positions

    return best_positions
# EVOLVE-BLOCK-END
