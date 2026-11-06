"""Dzugutov cluster global minima: structural search with polytetrahedral motifs and multi-scale optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 295.0  # seconds, safety margin before 330s kill
_lock = threading.Lock()


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
    if n < 2:
        return 0.0
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


def _get_icosahedral_points():
    """Return 12 vertices of a regular icosahedron, unit length."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_dodecahedral_points():
    """Return 20 vertices of a regular dodecahedron, unit length."""
    phi = (1 + np.sqrt(5)) / 2
    iphi = 1.0 / phi
    verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append([sx, sy, sz])
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            verts.append([0, sx * iphi, sy * phi])
            verts.append([sx * iphi, sy * phi, 0])
            verts.append([sx * phi, 0, sy * iphi])
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_cuboctahedral_points():
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


def _get_tetrahedral_points():
    """Return 4 vertices of a regular tetrahedron, unit length."""
    verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _build_mackay_icosahedron(n_atoms, nn_dist=1.0):
    """
    Build a Mackay icosahedral cluster.
    Shell sizes: 1, 12, 42, 92, 150, 220
    """
    if n_atoms <= 0:
        return np.array([])
    
    ico_verts = _get_icosahedral_points()
    
    # Shell 0: center
    positions = [np.zeros(3)]
    count = 1
    
    if n_atoms <= 1:
        return np.array(positions[:n_atoms])
    
    # Shell 1: 12 icosahedral vertices
    if count + 12 <= n_atoms:
        for v in ico_verts:
            positions.append(nn_dist * v)
        count += 12
    else:
        num_needed = n_atoms - count
        for i in range(num_needed):
            positions.append(nn_dist * ico_verts[i])
        count += num_needed
        return np.array(positions[:n_atoms])
    
    if count >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Shell 2: 42 points (12 dodecahedral + 30 edge midpoints)
    dod_verts = _get_dodecahedral_points()
    
    # Find edges of icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(ico_verts[i] - ico_verts[j])
    
    edge_set = set()
    for i in range(12):
        neighbors = np.argsort(dists[i])
        for j in neighbors[1:6]:
            if i < j:
                edge_set.add((i, j))
            else:
                edge_set.add((j, i))
    
    edge_mids = []
    for (i, j) in sorted(edge_set):
        mid = ico_verts[i] + ico_verts[j]
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    
    # Shell 2 radius: sqrt(5+2*sqrt(5))/2
    r5 = np.sqrt(5 + 2 * np.sqrt(5))
    shell2_radius = r5 / 2.0 * nn_dist
    
    # Add 12 dodecahedral points
    if count + 12 <= n_atoms:
        for v in dod_verts:
            positions.append(shell2_radius * v)
        count += 12
    else:
        num_needed = n_atoms - count
        for i in range(num_needed):
            positions.append(shell2_radius * dod_verts[i])
        count += num_needed
        return np.array(positions[:n_atoms])
    
    if count >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Add 30 edge midpoints
    if count + 30 <= n_atoms:
        for v in edge_mids:
            positions.append(shell2_radius * v)
        count += 30
    else:
        num_needed = n_atoms - count
        for i in range(num_needed):
            positions.append(shell2_radius * edge_mids[i])
        count += num_needed
        return np.array(positions[:n_atoms])
    
    if count >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Shell 3: 92 points - approximate with icosahedral directions at larger radius
    shell3_radius = r5 * nn_dist
    
    # Generate 92 directions: use icosahedral + dodecahedral + extra
    shell3_dirs = []
    for v in ico_verts:
        shell3_dirs.append(v)
    for v in dod_verts:
        shell3_dirs.append(v)
    for v in edge_mids:
        shell3_dirs.append(v)
    
    # Need 92 total, we have 12+20+30=62, add 30 more
    rng = np.random.default_rng(42)
    for _ in range(30):
        theta = np.arccos(2 * rng.random() - 1)
        phi_a = 2 * np.pi * rng.random()
        d = np.array([np.sin(theta) * np.cos(phi_a),
                      np.sin(theta) * np.sin(phi_a),
                      np.cos(theta)])
        shell3_dirs.append(d)
    
    shell3_dirs = np.array(shell3_dirs[:92])
    
    if count + 92 <= n_atoms:
        for v in shell3_dirs:
            positions.append(shell3_radius * v)
        count += 92
    else:
        num_needed = n_atoms - count
        for i in range(num_needed):
            positions.append(shell3_radius * shell3_dirs[i])
        count += num_needed
    
    return np.array(positions[:n_atoms])


def _build_polytetrahedral(n_atoms, nn_dist=1.0, twist_angle=0.0, seed=0):
    """
    Build a polytetrahedral cluster with twist between tetrahedra.
    This is a structural motif favored by the Dzugutov potential.
    """
    rng = np.random.default_rng(seed)
    
    tetra_verts = _get_tetrahedral_points()
    
    # Rotation matrices for twist
    def rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)
    
    positions = [np.zeros(3)]
    count = 1
    
    # Build tetrahedral units around center
    # Each tetrahedron has 4 vertices, but we share the center
    # So each tetrahedron adds 3 new atoms
    
    # Start with a central tetrahedron (4 atoms, center + 3 others)
    # Actually, let's build it as nested tetrahedra
    
    # Layer 1: 4 tetrahedra around center, each sharing the center
    # This gives 1 + 4*3 = 13 atoms (but some sharing)
    
    # Simpler approach: build a BCC-like structure with tetrahedral motifs
    
    # Use icosahedral directions for the outer shell, but with tetrahedral sub-clusters
    
    ico_verts = _get_icosahedral_points()
    
    # For small clusters, use compact polytetrahedral arrangements
    if n_atoms <= 13:
        # Mackay-like icosahedron
        if n_atoms == 1:
            return np.array(positions)
        if n_atoms <= 13:
            # Center + up to 12 icosahedral vertices
            num_outer = min(n_atoms - 1, 12)
            for i in range(num_outer):
                positions.append(nn_dist * ico_verts[i])
            return np.array(positions[:n_atoms])
    
    # For larger clusters, build polytetrahedral shells
    # Use a combination of icosahedral and tetrahedral directions
    
    all_dirs = list(ico_verts)
    
    # Add tetrahedral directions at different orientations
    for k in range(4):
        rot = rotation_matrix(np.array([k%2, (k//2)%2, k%3 if k>1 else 0]), twist_angle + k * np.pi/4)
        for v in tetra_verts:
            all_dirs.append(rot @ v)
    
    # Add more directions for larger clusters
    while len(all_dirs) < n_atoms:
        theta = np.arccos(2 * rng.random() - 1)
        phi_a = 2 * np.pi * rng.random()
        d = np.array([np.sin(theta) * np.cos(phi_a),
                      np.sin(theta) * np.sin(phi_a),
                      np.cos(theta)])
        all_dirs.append(d)
    
    all_dirs = np.array(all_dirs[:n_atoms])
    
    # Assign radii: inner atoms closer, outer atoms farther
    for i, v in enumerate(all_dirs):
        if i == 0:
            positions.append(np.zeros(3))
        else:
            # Radius depends on shell
            shell = i // 12 + 1
            r = nn_dist * (1.0 + 0.8 * (shell - 1))
            positions.append(r * v)
    
    return np.array(positions[:n_atoms])


def _build_fcc_cluster(n_atoms, nn_dist=1.0):
    """Build an FCC cluster seed."""
    positions = [np.zeros(3)]
    count = 1
    
    # FCC lattice points
    a = nn_dist / np.sqrt(2)
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if count >= n_atoms:
                    break
                # FCC: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                for offset in [[0,0,0], [0,a/2,a/2], [a/2,0,a/2], [a/2,a/2,0]]:
                    p = np.array([i*a + offset[0], j*a + offset[1], k*a + offset[2]])
                    if np.linalg.norm(p) < 3.0 * nn_dist:
                        positions.append(p)
                        count += 1
                        if count >= n_atoms:
                            break
                if count >= n_atoms:
                    break
        if count >= n_atoms:
            break
    
    return np.array(positions[:n_atoms])


def _scatter_compact(n, radius=3.0, seed=0):
    """Generate a random compact cluster with minimum distance."""
    rng = np.random.default_rng(seed)
    points = []
    max_attempts = 50000
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
    
    # If not enough, add more with smaller radius
    while len(points) < n:
        p = rng.uniform(-0.4, 0.4, 3)
        points.append(p)
    
    return np.array(points[:n])


def _generate_seeds(n, seed_base=0):
    """Generate diverse seed structures."""
    seeds = []
    
    # 1. Mackay icosahedron
    try:
        mackay = _build_mackay_icosahedron(n, nn_dist=1.0)
        if len(mackay) == n and np.all(np.isfinite(mackay)):
            seeds.append(mackay)
    except:
        pass
    
    # 2. Mackay with different nn_dist
    try:
        mackay2 = _build_mackay_icosahedron(n, nn_dist=0.95)
        if len(mackay2) == n and np.all(np.isfinite(mackay2)):
            seeds.append(mackay2)
    except:
        pass
    
    # 3. Polytetrahedral with twist
    for twist in [0.0, np.pi/6, np.pi/4, np.pi/3]:
        try:
            pt = _build_polytetrahedral(n, nn_dist=1.0, twist_angle=twist, seed=seed_base)
            if len(pt) == n and np.all(np.isfinite(pt)):
                seeds.append(pt)
        except:
            pass
    
    # 4. FCC cluster
    try:
        fcc = _build_fcc_cluster(n, nn_dist=1.0)
        if len(fcc) == n and np.all(np.isfinite(fcc)):
            seeds.append(fcc)
    except:
        pass
    
    # 5-8. Random compact seeds
    for trial in range(4):
        try:
            sc = _scatter_compact(n, radius=2.5 + 0.5 * trial, seed=seed_base + 100 + trial)
            seeds.append(sc)
        except:
            pass
    
    return seeds


def _l_bfgs_refine(pos, max_iter=100, time_limit=10):
    """Refine using L-BFGS with gradient."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        p = x.reshape(-1, 3)
        if np.any(np.isfinite(p) == False):
            return 1e10, np.zeros_like(x)
        try:
            e = _energy(p)
            if e >= 1e9:
                return 1e10, np.zeros_like(x)
            grad = dzugutov_energy_gradient(p)
            if np.any(np.isfinite(grad) == False):
                return 1e10, np.zeros_like(x)
            return e, grad.flatten()
        except:
            return 1e10, np.zeros_like(x)
    
    try:
        from scipy.optimize import minimize
        x0 = pos.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'maxfun': 500})
        if result.success or result.fun < 1e9:
            return result.fun, result.x.reshape(-1, 3)
        else:
            return _energy(pos), pos
    except:
        return _energy(pos), pos


def _kick_and_relax(pos, step_size, rng, time_limit=5):
    """Perturb and relax."""
    start_time = time.time()
    perturbed = pos + rng.normal(0, step_size, size=pos.shape)
    
    try:
        e, relaxed = relax(perturbed, 'dzugutov', None)
        if not np.all(np.isfinite(e)) or e >= 1e9:
            return None, None
        if _energy(relaxed) >= 1e9:
            return None, None
        return e, relaxed
    except:
        return None, None


def _basin_hopping(pos, n, max_iter=80, time_limit=30, rng_seed=None):
    """
    Basin hopping with multi-scale kicks and L-BFGS refinement.
    """
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
        
        # Multi-scale step sizes
        progress = it / max_iter
        base_step = 0.3 + 0.4 * (1 - progress)
        
        # Occasionally take a larger kick
        if rng.random() < 0.15:
            step_size = base_step * 2.5
        elif rng.random() < 0.3:
            step_size = base_step * 1.5
        else:
            step_size = base_step
        
        e, relaxed = _kick_and_relax(pos, step_size, rng, time_limit=5)
        
        if e is None:
            continue
        
        # L-BFGS refinement
        e_refined, pos_refined = _l_bfgs_refine(relaxed, max_iter=50, time_limit=5)
        if e_refined < e:
            e = e_refined
            relaxed = pos_refined
        
        if e < current_energy:
            current_energy = e
            pos = relaxed
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        else:
            # Metropolis-like acceptance
            delta_e = e - current_energy
            if delta_e < 0.5 and rng.random() < np.exp(-delta_e / 0.2):
                pos = relaxed
    
    return best_energy, best_pos


def _simulated_annealing(pos, n, max_iter=100, time_limit=20, rng_seed=None):
    """
    Simulated annealing with L-BFGS refinement.
    """
    start_time = time.time()
    current_energy = _energy(pos)
    best_energy = current_energy
    best_pos = pos.copy()
    
    if rng_seed is None:
        rng_seed = hash(tuple(pos.flatten()[:3])) % (2**32)
    rng = np.random.default_rng(rng_seed)
    
    T_start = 1.0
    T_end = 0.01
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if _remaining() < 5:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        step_size = 0.3 + 0.3 * (1 - it / max_iter)
        
        e, relaxed = _kick_and_relax(pos, step_size, rng, time_limit=5)
        
        if e is None:
            continue
        
        e_refined, pos_refined = _l_bfgs_refine(relaxed, max_iter=30, time_limit=5)
        if e_refined < e:
            e = e_refined
            relaxed = pos_refined
        
        delta_e = e - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_energy = e
            pos = relaxed
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    return best_energy, best_pos


def _single_search(n, seed_idx, time_limit=35, seed_base=0):
    """Run a single search from a given seed."""
    start_time = time.time()
    seeds = _generate_seeds(n, seed_base=seed_base)
    
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
    
    # L-BFGS refinement
    e_ref, pos_ref = _l_bfgs_refine(pos, max_iter=100, time_limit=10)
    if e_ref < best_energy:
        best_energy = e_ref
        best_pos = pos_ref
    
    # Basin hopping
    if _remaining() > 10:
        try:
            bh_energy, bh_pos = _basin_hopping(pos_ref, n, max_iter=60, time_limit=min(20, _remaining() - 10),
                                               rng_seed=seed_base * 1000 + seed_idx)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            pass
    
    # Simulated annealing if time permits
    if _remaining() > 10 and seed_idx % 2 == 0:
        try:
            sa_energy, sa_pos = _simulated_annealing(pos_ref, n, max_iter=50, time_limit=min(15, _remaining() - 10),
                                                     rng_seed=seed_base * 2000 + seed_idx)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pos = sa_pos
        except:
            pass
    
    return best_energy, best_pos


def _fix_minimum_distance(pos, min_dist=0.5):
    """Ensure all atoms are at least min_dist apart."""
    n = len(pos)
    if n < 2:
        return pos
    
    pos = pos.copy()
    for _ in range(200):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_d = np.min(dist)
        
        if min_d >= min_dist:
            break
        
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if i > j:
            i, j = j, i
        
        dir_vec = pos[i] - pos[j]
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-10:
            dir_vec = np.array([1.0, 0, 0])
        else:
            dir_vec = dir_vec / norm
        
        pos[i] += 0.01 * dir_vec
        pos[j] -= 0.01 * dir_vec
    
    return pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    
    # Generate seeds
    seeds = _generate_seeds(n, seed_base=0)
    
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
            best_pos = _scatter_compact(n, radius=2.0, seed=999)
            try:
                _, best_pos = relax(best_pos, 'dzugutov', None)
            except:
                pass
    
    # Final L-BFGS refinement
    if best_pos is not None and _remaining() > 5:
        try:
            e_ref, pos_ref = _l_bfgs_refine(best_pos, max_iter=100, time_limit=min(15, _remaining() - 5))
            if e_ref < best_energy:
                best_energy = e_ref
                best_pos = pos_ref
        except:
            pass
    
    # Fix minimum distance
    if best_pos is not None:
        best_pos = _fix_minimum_distance(best_pos, min_dist=0.5)
        try:
            _, best_pos = relax(best_pos, 'dzugutov', None)
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
