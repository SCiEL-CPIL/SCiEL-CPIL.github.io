"""Dzugutov cluster global minima search with hierarchical core seeding and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
_TOTAL_BUDGET = 310.0  # 330s limit with 20s safety margin
_HARD_LIMIT = 325.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _hard_remaining():
    return _HARD_LIMIT - _elapsed()

def _dzugutov_energy(positions):
    """Vectorized Dzugutov energy calculation."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Mask to avoid self-interaction and double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dists[mask]
    
    total = 0.0
    mask_a = dists < a_DZ
    mask_b = (dists >= a_DZ) & (dists < b_DZ)
    
    if np.any(mask_a):
        da = dists[mask_a]
        da = np.maximum(da, 1e-10)
        v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
        total += np.sum(v_a)
        
    if np.any(mask_b):
        db = dists[mask_b]
        db = np.maximum(db, 1e-10)
        v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
        total += np.sum(v_b)
        
    return total

def _generate_mackay_icosahedron(n_atoms, rng):
    """Generate a Mackay icosahedron cluster."""
    if n_atoms == 13:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        edge = 2 * phi
        verts = verts / edge * 1.15
        return np.vstack([[0.0, 0.0, 0.0], verts])
    
    # For larger clusters, use nested icosahedra or FCC
    # Simple fallback: FCC cluster
    return _generate_fcc_cluster(n_atoms, rng)

def _generate_fcc_cluster(n_atoms, rng):
    """Generate an FCC cluster."""
    points = []
    a = 1.1
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0) / 2)) + 2
    
    # Generate FCC points
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([ix + dx, iy + dy, iz + dz]) * a
                    points.append(p)
                    if len(points) >= n_atoms:
                        break
                if len(points) >= n_atoms:
                    break
            if len(points) >= n_atoms:
                break
        if len(points) >= n_atoms:
            break
    
    points = np.array(points[:n_atoms])
    # Sort by distance from center to ensure compactness
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n_atoms]]

def _generate_random_compact(n_atoms, rng):
    """Generate a random compact cluster."""
    radius = 1.2 * n_atoms ** (1.0 / 3.0)
    points = []
    min_dist = 0.6
    
    attempts = 0
    while len(points) < n_atoms and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
            
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _basin_hopping(positions, rng, max_iter=50, step_size=0.4, time_limit=10.0):
    """Basin hopping with adaptive step size."""
    start_time = time.time()
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        perturbed = current + rng.normal(0, step_size, current.shape)
        
        try:
            e_new, relaxed = relax(perturbed, 'dzugutov', None)
        except Exception:
            continue
            
        # Metropolis acceptance
        if e_new < current_energy or rng.random() < np.exp(-(e_new - current_energy) / 0.1):
            current = relaxed
            current_energy = e_new
            
        step_size *= 0.98
        
    return current, current_energy

def _worker_basin_hopping(size, seed, strategy, time_limit):
    """Worker: generate seed and run basin hopping."""
    rng = np.random.default_rng(seed)
    
    if strategy == 'mackay':
        cand = _generate_mackay_icosahedron(size, rng)
    elif strategy == 'fcc':
        cand = _generate_fcc_cluster(size, rng)
    else:
        cand = _generate_random_compact(size, rng)
        
    cand += rng.normal(0, 0.05, cand.shape)
    
    try:
        # Initial relaxation
        e_init, relaxed_init = relax(cand, 'dzugutov', None)
        
        # Basin hopping
        bh_pos, bh_energy = _basin_hopping(
            relaxed_init, 
            rng, 
            max_iter=30, 
            step_size=0.4, 
            time_limit=time_limit
        )
        
        return bh_energy, bh_pos
    except Exception:
        return np.inf, cand

def _process_instance(size, potential, parameter, time_budget):
    """Process a single instance with parallel basin hopping."""
    
    # Determine number of parallel chains based on time budget
    n_chains = 4 if time_budget > 30 else 2
    
    strategies = ['mackay', 'fcc', 'random', 'random']
    
    best_energy = np.inf
    best_positions = None
    
    # If time is very short, just do a single relaxation
    if time_budget < 10:
        cand = _generate_mackay_icosahedron(size, np.random.default_rng(size))
        try:
            e, pos = relax(cand, 'dzugutov', None)
            return pos
        except:
            return cand

    time_per_chain = (time_budget - 5.0) / n_chains
    
    with ThreadPoolExecutor(max_workers=n_chains) as executor:
        futures = []
        for i in range(n_chains):
            strategy = strategies[i % len(strategies)]
            seed = size * 1000 + i * 7
            futures.append(executor.submit(_worker_basin_hopping, size, seed, strategy, time_per_chain))
            
        for future in as_completed(futures):
            if _hard_remaining() < 5.0:
                break
            try:
                energy, positions = future.result(timeout=time_per_chain + 5)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # Final safety
    if best_positions is None or len(best_positions) != size:
        best_positions = _generate_fcc_cluster(size, np.random.default_rng(size))
        
    return best_positions

def construct(instance):
    """Find low-energy Dzugutov cluster."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    remaining_instances = [s for s in _INSTANCE_ORDER if s >= size]
    time_per_instance = _hard_remaining() / max(1, len(remaining_instances))
    time_per_instance = min(time_per_instance, 50.0)
    
    if size in _INSTANCE_RESULTS:
        return _INSTANCE_RESULTS[size]
        
    result = _process_instance(size, potential, parameter, time_per_instance)
    
    _INSTANCE_RESULTS[size] = result
    
    return result
# EVOLVE-BLOCK-END
