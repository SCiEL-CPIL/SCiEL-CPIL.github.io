"""Dzugutov cluster global minima: parallel simulated annealing with polytetrahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 315.0  # safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _build_icosahedral_points():
    """Generate the 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm = np.linalg.norm(verts[0])
    return verts / norm

def _generate_polytetrahedral_seed(n, rng, scale=1.0):
    """
    Generate a polytetrahedral seed: place icosahedra sharing vertices.
    """
    ico_verts = _build_icosahedral_points() * scale
    
    if n <= 13:
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        if n == 2:
            return np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0]])
        points = [np.zeros(3)]
        for i in range(min(n - 1, 12)):
            points.append(ico_verts[i])
        return np.array(points)
    
    points = [np.zeros(3)]
    for i in range(12):
        points.append(ico_verts[i].copy())
    
    # Add more icosahedra by placing their centers at existing vertex positions
    centers = [np.zeros(3)]
    n_icos_needed = (n - 13) // 12 + 1
    
    for i in range(1, min(n_icos_needed, 12)):
        center_pos = ico_verts[i].copy()
        centers.append(center_pos)
        for j in range(12):
            new_point = center_pos + ico_verts[j]
            is_dup = False
            for p in points:
                if np.linalg.norm(p - new_point) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                points.append(new_point)
    
    while len(points) < n:
        idx = rng.integers(len(points))
        base = points[idx]
        offset = rng.normal(0, 0.5, 3)
        new_point = base + offset
        is_dup = False
        for p in points:
            if np.linalg.norm(p - new_point) < 0.5:
                is_dup = True
                break
        if not is_dup:
            points.append(new_point)
    
    return np.array(points[:n])

def _enforce_min_distance(positions, min_dist=0.5, max_iters=30):
    """Enforce minimum distance between atoms."""
    n = len(positions)
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    diff = positions[i] - positions[j]
                    if d < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff /= d
                    push = (min_dist - d) / 2 + 0.01
                    positions[i] += diff * push
                    positions[j] -= diff * push
                    moved = True
        if not moved:
            break
    return positions

def _generate_compact_random(n, rng, scale=1.0):
    """Generate a compact random seed."""
    r_max = scale * n ** (1/3) * 0.5
    positions = rng.uniform(-r_max, r_max, (n, 3))
    return _enforce_min_distance(positions)

def _generate_bcc_seed(n, rng, scale=1.0):
    """Generate a BCC-like seed."""
    a = scale * 2.0
    positions = []
    n_side = int(np.ceil(n ** (1/3)))
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                pos = np.array([i * a, j * a, k * a])
                positions.append(pos)
                if len(positions) < n:
                    pos_center = pos + np.array([a/2, a/2, a/2])
                    positions.append(pos_center)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    return _enforce_min_distance(positions)

def _generate_fcc_seed(n, rng, scale=1.0):
    """Generate an FCC-like seed."""
    a = scale * 1.8
    positions = []
    n_side = int(np.ceil((n * 0.75) ** (1/3)))
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                pos = np.array([i * a, j * a, k * a])
                positions.append(pos)
                # Face centers
                if i < n_side - 1 and len(positions) < n:
                    positions.append(pos + np.array([a/2, a/2, 0]))
                if j < n_side - 1 and len(positions) < n:
                    positions.append(pos + np.array([a/2, 0, a/2]))
                if k < n_side - 1 and len(positions) < n:
                    positions.append(pos + np.array([0, a/2, a/2]))
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    return _enforce_min_distance(positions)

def _simulated_annealing(positions, potential, rng, max_iters=100, time_check=None):
    """
    Simulated Annealing with adaptive temperature.
    """
    current_positions = positions.copy()
    current_energy, _ = relax(positions, potential)
    
    best_positions = current_positions.copy()
    best_energy = current_energy
    
    # Adaptive parameters
    T_init = 2.0
    T_final = 0.01
    cooling_rate = (T_final / T_init) ** (1.0 / max(1, max_iters))
    step_scale = 0.3
    
    for it in range(max_iters):
        if time_check is not None and time_check() < 15:
            break
            
        T = T_init * (cooling_rate ** it)
        
        # Random perturbation
        step = step_scale * rng.normal(0, 1, current_positions.shape)
        new_positions = current_positions + step
        
        # Relax
        energy, relaxed = relax(new_positions, potential)
        
        # Metropolis acceptance
        if energy < current_energy:
            current_positions = relaxed.copy()
            current_energy = energy
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        else:
            delta_e = energy - current_energy
            if T > 1e-10 and rng.random() < np.exp(-delta_e / T):
                current_positions = relaxed.copy()
                current_energy = energy
                
        # Occasionally reheat
        if it % 20 == 19 and it < max_iters - 5:
            T_init = max(T_init * 0.5, T_final * 10)
            cooling_rate = (T_final / T_init) ** (1.0 / max(1, max_iters - it))
            
    return best_positions, best_energy

def _worker_sa(args):
    """Worker function for parallel SA."""
    positions, potential, seed, max_iters, deadline = args
    rng = np.random.default_rng(seed)
    
    _START_TIME_WORKER = time.time()
    
    def time_check():
        return deadline - (time.time() - _START_TIME_WORKER)
    
    best_positions, best_energy = _simulated_annealing(
        positions, potential, rng,
        max_iters=max_iters, time_check=time_check
    )
    
    return best_positions, best_energy

def construct(instance):
    """
    Find low-energy Dzugutov clusters using parallel simulated annealing.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_generate_polytetrahedral_seed(n, rng, scale=1.0))
    seeds.append(_generate_compact_random(n, rng, scale=1.0))
    seeds.append(_generate_bcc_seed(n, rng, scale=1.0))
    seeds.append(_generate_fcc_seed(n, rng, scale=1.0))
    seeds.append(_generate_polytetrahedral_seed(n, rng, scale=1.05))
    seeds.append(_generate_compact_random(n, rng, scale=0.95))
    
    # Relax all seeds and find best
    for seed in seeds:
        seed = _enforce_min_distance(seed)
        energy, positions = relax(seed, potential)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Parallel simulated annealing from multiple seeds
    time_remaining = _get_time_remaining()
    
    if best_positions is not None and time_remaining > 40:
        n_workers = min(4, mp.cpu_count())
        time_per_worker = (time_remaining - 20) / n_workers
        
        # Prepare work items
        work_items = []
        for i in range(n_workers):
            if i < len(seeds):
                seed_pos = seeds[i].copy()
            else:
                seed_pos = best_positions.copy()
            
            seed_val = n * 1000 + i * 7919
            max_iters = int(time_per_worker / 0.3)  # estimate iterations
            max_iters = max(20, min(150, max_iters))
            
            work_items.append((seed_pos, potential, seed_val, max_iters, time_remaining - 20))
        
        try:
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(_worker_sa, work_items)
                
                for pos, energy in results:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos.copy()
        except Exception as e:
            # Fallback to sequential if parallel fails
            pass
    
    # Final sequential SA from best position if time remains
    time_remaining = _get_time_remaining()
    if best_positions is not None and time_remaining > 25:
        rng2 = np.random.default_rng(n * 31337 + 99)
        max_iters = min(80, int((time_remaining - 10) / 0.3))
        max_iters = max(10, max_iters)
        
        def time_check():
            return _get_time_remaining()
        
        final_positions, final_energy = _simulated_annealing(
            best_positions, potential, rng2,
            max_iters=max_iters, time_check=time_check
        )
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        best_positions = np.array([[i * 1.1, 0.0, 0.0] for i in range(n)])
        
    return best_positions
# EVOLVE-BLOCK-END
