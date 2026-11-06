"""Dzugutov cluster global minima: improved icosahedral seeds + multi-scale basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 295.0  # seconds, safety margin before 330s kill


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    triu_idx = np.triu_indices(n, k=1)
    r = dist[triu_idx]
    
    if np.min(r) < 0.5:
        return 1e10
    
    E = 0.0
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        E += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2 = r[mask2]
        E += np.sum(B * np.exp(d / (r2 - b)))
    
    return float(E)


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, unit length."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _dodecahedron_vertices():
    """Return 20 vertices of a regular dodecahedron centered at origin, unit length."""
    phi = (1 + np.sqrt(5)) / 2
    iphi = 1.0 / phi
    verts = []
    # 8 cube vertices
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append([sx, sy, sz])
    # 12 vertices from rectangle permutations
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            verts.append([0, sx * iphi, sy * phi])
            verts.append([sx * iphi, sy * phi, 0])
            verts.append([sx * phi, 0, sy * iphi])
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _cuboctahedron_vertices():
    """Return 12 vertices of a cuboctahedron, unit length."""
    verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            verts.append([sx, sy, 0])
            verts.append([sx, 0, sy])
            verts.append([0, sx, sy])
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _icosahedral_shell_directions(shell):
    """
    Generate directions for icosahedral shells.
    shell=1: 12 vertices
    shell=2: 42 (12 face centers + 30 edge midpoints)
    shell=3: 92
    """
    v = _icosahedron_vertices()
    
    if shell == 1:
        return v
    
    if shell == 2:
        # Find edges (nearest neighbor pairs)
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                dists[i, j] = np.linalg.norm(v[i] - v[j])
        
        edge_set = set()
        for i in range(12):
            neighbors = np.argsort(dists[i])
            for j in neighbors[1:6]:
                if i < j:
                    edge_set.add((i, j))
                else:
                    edge_set.add((j, i))
        
        # Edge midpoints (30 directions)
        edge_mids = []
        for (i, j) in edge_set:
            mid = v[i] + v[j]
            mid = mid / np.linalg.norm(mid)
            edge_mids.append(mid)
        
        # Face centers (20 directions) - find triangles
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                if (i, j) in edge_set:
                    for k in range(j + 1, 12):
                        if (i, k) in edge_set and (j, k) in edge_set:
                            fc = v[i] + v[j] + v[k]
                            fc = fc / np.linalg.norm(fc)
                            face_centers.append(fc)
        
        # We have 30 edge mids + 20 face centers = 50, but we need 42
        # The 42-atom shell: 12 dodecahedral + 30 edge midpoints
        # Use 12 of the face centers (or dodecahedron vertices) + 30 edge mids
        # For now, use 30 edge mids + 12 face centers
        result = np.array(edge_mids[:30] + face_centers[:12])
        return result
    
    if shell == 3:
        # 92 atoms: more complex. Use a combination of directions
        # For now, generate 92 directions from a dense icosahedral pattern
        # Use the 42 from shell 2 plus additional points
        shell2 = _icosahedral_shell_directions(2)
        # Add 50 more from a Fibonacci sphere with icosahedral bias
        rng = np.random.default_rng(42)
        extra = []
        for _ in range(50):
            # Random direction
            theta = np.arccos(2 * rng.random() - 1)
            phi_a = 2 * np.pi * rng.random()
            d = np.array([np.sin(theta) * np.cos(phi_a),
                          np.sin(theta) * np.sin(phi_a),
                          np.cos(theta)])
            extra.append(d)
        result = np.concatenate([shell2, np.array(extra)], axis=0)
        return result[:92]
    
    return v


def _build_icosahedral(n_atoms, nn_dist=1.0):
    """
    Build a Mackay icosahedral cluster with n_atoms atoms.
    Shell sizes: 1, 12, 42, 92, 150, 220
    Radii: 0, 1, sqrt(5+2*sqrt(5))/2 ≈ 1.902, sqrt(5+2*sqrt(5)) ≈ 3.804
    """
    shell_sizes = [1, 12, 42, 92, 150, 220]
    # Radii for Mackay icosahedron (in units of nearest neighbor distance)
    # Shell 0: r=0
    # Shell 1: r=1
    # Shell 2: r=sqrt(5+2*sqrt(5))/2
    # Shell 3: r=sqrt(5+2*sqrt(5))
    # Shell 4: r=3*sqrt(5+2*sqrt(5))/2
    r5 = np.sqrt(5 + 2 * np.sqrt(5))
    shell_radii = [0.0, 1.0, r5 / 2.0, r5, 3.0 * r5 / 2.0, 2.0 * r5]
    
    positions = [np.zeros(3)]
    count = 1
    
    for shell_idx in range(1, 6):
        shell_size = shell_sizes[shell_idx]
        shell_radius = shell_radii[shell_idx] * nn_dist
        
        if count + shell_size > n_atoms:
            # Only add partial shell
            num_needed = n_atoms - count
            if shell_idx <= 3:
                dirs = _icosahedral_shell_directions(shell_idx)
                for i in range(num_needed):
                    positions.append(shell_radius * dirs[i])
            else:
                # For larger shells, use random directions
                rng = np.random.default_rng(100 + shell_idx)
                for i in range(num_needed):
                    theta = np.arccos(2 * rng.random() - 1)
                    phi_a = 2 * np.pi * rng.random()
                    d = np.array([np.sin(theta) * np.cos(phi_a),
                                  np.sin(theta) * np.sin(phi_a),
                                  np.cos(theta)])
                    positions.append(shell_radius * d)
            count += num_needed
            break
        else:
            if shell_idx <= 3:
                dirs = _icosahedral_shell_directions(shell_idx)
            else:
                rng = np.random.default_rng(100 + shell_idx)
                dirs = []
                for i in range(shell_size):
                    theta = np.arccos(2 * rng.random() - 1)
                    phi_a = 2 * np.pi * rng.random()
                    d = np.array([np.sin(theta) * np.cos(phi_a),
                                  np.sin(theta) * np.sin(phi_a),
                                  np.cos(theta)])
                    dirs.append(d)
                dirs = np.array(dirs)
            
            for i in range(shell_size):
                positions.append(shell_radius * dirs[i])
            count += shell_size
    
    return np.array(positions[:n_atoms])


def _build_polyhedral(n_atoms, poly_type='icosahedron', nn_dist=1.0):
    """Build a polyhedral cluster seed."""
    if poly_type == 'icosahedron':
        return _build_icosahedral(n_atoms, nn_dist)
    elif poly_type == 'dodecahedron':
        # Use dodecahedron vertices for inner shell, then fill
        verts = _dodecahedron_vertices()
        positions = [np.zeros(3)]
        for v in verts:
            positions.append(nn_dist * v)
        count = 1 + len(verts)
        remaining = n_atoms - count
        if remaining > 0:
            rng = np.random.default_rng(7)
            for _ in range(remaining):
                theta = np.arccos(2 * rng.random() - 1)
                phi_a = 2 * np.pi * rng.random()
                d = np.array([np.sin(theta) * np.cos(phi_a),
                              np.sin(theta) * np.sin(phi_a),
                              np.cos(theta)])
                r = nn_dist * (1.5 + 0.3 * rng.random())
                positions.append(r * d)
        return np.array(positions[:n_atoms])
    elif poly_type == 'cuboctahedron':
        verts = _cuboctahedron_vertices()
        positions = [np.zeros(3)]
        for v in verts:
            positions.append(nn_dist * v)
        count = 1 + len(verts)
        remaining = n_atoms - count
        if remaining > 0:
            rng = np.random.default_rng(13)
            for _ in range(remaining):
                theta = np.arccos(2 * rng.random() - 1)
                phi_a = 2 * np.pi * rng.random()
                d = np.array([np.sin(theta) * np.cos(phi_a),
                              np.sin(theta) * np.sin(phi_a),
                              np.cos(theta)])
                r = nn_dist * (1.5 + 0.3 * rng.random())
                positions.append(r * d)
        return np.array(positions[:n_atoms])
    else:
        return _build_icosahedral(n_atoms, nn_dist)


def _scatter(n, radius=3.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-0.3, 0.3, 3)
        points.append(p)
    
    return np.array(points[:n])


def _seeds(n, seed_base=0):
    """Generate diverse seed structures."""
    seeds = []
    
    # 1. Icosahedral Mackay seed
    try:
        ico = _build_icosahedral(n, nn_dist=1.0)
        if len(ico) == n and np.all(np.isfinite(ico)):
            seeds.append(ico)
    except:
        pass
    
    # 2. Icosahedral with slightly different nn_dist
    try:
        ico2 = _build_icosahedral(n, nn_dist=0.95)
        if len(ico2) == n and np.all(np.isfinite(ico2)):
            seeds.append(ico2)
    except:
        pass
    
    # 3. Dodecahedral seed
    try:
        dod = _build_polyhedral(n, 'dodecahedron', nn_dist=1.0)
        if len(dod) == n and np.all(np.isfinite(dod)):
            seeds.append(dod)
    except:
        pass
    
    # 4. Cuboctahedral seed
    try:
        cub = _build_polyhedral(n, 'cuboctahedron', nn_dist=1.0)
        if len(cub) == n and np.all(np.isfinite(cub)):
            seeds.append(cub)
    except:
        pass
    
    # 5-7. Random compact seeds
    for trial in range(3):
        try:
            sc = _scatter(n, radius=2.5 + 0.5 * trial, seed=seed_base + trial)
            seeds.append(sc)
        except:
            pass
    
    return seeds


def _basin_hopping(pos, n, max_iter=60, time_limit=25, rng_seed=None):
    """Multi-scale basin hopping: perturb, relax, keep if lower energy."""
    start_time = time.time()
    current_energy = _energy(pos)
    best_energy = current_energy
    best_pos = pos.copy()
    
    if rng_seed is None:
        rng_seed = hash(tuple(pos.flatten()[:3])) % (2**32)
    rng = np.random.default_rng(rng_seed)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if _remaining() < 5:
            break
        
        # Multi-scale step sizes: start large, decrease
        base_step = 0.4 + 0.3 * (1 - it / max_iter)
        # Occasionally take a larger kick
        if rng.random() < 0.1:
            step_size = base_step * 2.0
        else:
            step_size = base_step
        
        perturbed = pos + rng.normal(0, step_size, size=pos.shape)
        
        # Relax
        try:
            e, relaxed = relax(perturbed, 'dzugutov', None)
        except:
            continue
        
        if not np.all(np.isfinite(e)) or e >= 1e9:
            continue
        
        if _energy(relaxed) >= 1e9:
            continue
        
        if e < current_energy:
            current_energy = e
            pos = relaxed
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        else:
            # Metropolis-like acceptance for exploration
            if rng.random() < np.exp(-0.5 * (e - current_energy)):
                pos = relaxed
    
    return best_energy, best_pos


def _single_search(n, seed_idx, time_limit=35, seed_base=0):
    """Run a single search from a given seed."""
    start_time = time.time()
    seeds = _seeds(n, seed_base=seed_base)
    
    if seed_idx >= len(seeds):
        seed_idx = seed_idx % len(seeds)
    
    pos = seeds[seed_idx].copy()
    
    # First, relax the seed
    try:
        e, pos = relax(pos, 'dzugutov', None)
    except:
        return 1e10, None
    
    if not np.all(np.isfinite(e)) or e >= 1e9:
        return 1e10, None
    
    best_energy = e
    best_pos = pos.copy()
    
    # Basin hopping
    try:
        bh_energy, bh_pos = _basin_hopping(pos, n, max_iter=50, time_limit=time_limit,
                                           rng_seed=seed_base * 1000 + seed_idx)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    except:
        pass
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    
    # Generate seeds
    seeds = _seeds(n, seed_base=0)
    
    best_energy = 1e10
    best_pos = None
    
    # Run parallel searches
    n_seeds = len(seeds)
    n_threads = min(4, n_seeds)
    
    # Time budget per search
    total_time_budget = min(280.0, _remaining())
    per_search_time = total_time_budget / n_threads
    
    results = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {}
        for i in range(n_seeds):
            if _remaining() < 10:
                break
            future = executor.submit(_single_search, n, i, per_search_time, seed_base=i)
            futures[future] = i
        
        for future in as_completed(futures):
            if _remaining() < 5:
                break
            try:
                energy, pos = future.result(timeout=15)
                if pos is not None and energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except:
                pass
    
    # If no valid result, use a fallback
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        if seeds:
            try:
                e, pos = relax(seeds[0], 'dzugutov', None)
                if np.all(np.isfinite(pos)) and e < 1e9:
                    best_pos = pos
                    best_energy = e
            except:
                pass
        
        if best_pos is None:
            best_pos = _scatter(n, radius=2.0, seed=999)
            try:
                _, best_pos = relax(best_pos, 'dzugutov', None)
            except:
                pass
    
    # Final check: ensure minimum distance
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        if min_dist < 0.5:
            for _ in range(200):
                if min_dist >= 0.5:
                    break
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                if i > j:
                    i, j = j, i
                dir_vec = best_pos[i] - best_pos[j]
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-10:
                    dir_vec = np.array([1.0, 0, 0])
                else:
                    dir_vec = dir_vec / norm
                best_pos[i] += 0.01 * dir_vec
                best_pos[j] -= 0.01 * dir_vec
                diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
            
            try:
                _, best_pos = relax(best_pos, 'dzugutov', None)
            except:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
