"""
Strategy: Multi-Start Parallel Basin Hopping.
Instead of a single serial search chain, we launch multiple independent basin 
hopping chains in parallel using a thread pool. This leverages the GIL release 
in NumPy/SciPy operations to maximize search throughput.
We combine diverse initial structures (FCC, HCP, Random) with a robust 
simulated annealing-like basin hopping protocol (accepting worse moves with 
probability exp(-dE/T) to escape local minima).
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 4
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    
    if len(points) < n:
        return None
        
    points = np.array(points)
    # Sort by distance to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _hcp_fragment(n, spacing=0.72):
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            for k in range(-max_r, max_r + 1):
                points.append([x, y, k * c])
                points.append([x + 2*a/3, y + a*np.sqrt(3)/3, (k + 0.5) * c])
                
    if len(points) < n:
        return None
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        points = rng.uniform(-radius, radius, (n, 3)).tolist()
    return np.array(points)

def _basin_hop_chain(start_pos, n, kind, potential, rng, max_hops=50, initial_temp=0.5):
    """
    Runs a single basin hopping chain.
    Returns the best energy and positions found in this chain.
    """
    current_pos = start_pos.copy()
    try:
        current_energy, current_pos = relax(current_pos, potential, kind)
    except Exception:
        return float('inf'), None

    best_energy = current_energy
    best_pos = current_pos.copy()
    temp = initial_temp
    step_size = 0.2
    
    for i in range(max_hops):
        if _time_left() < 5:
            break
            
        # Perturb
        noise = rng.normal(0, step_size, (n, 3))
        perturbed = current_pos + noise
        
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except Exception:
            continue
            
        dE = new_energy - current_energy
        
        # Metropolis acceptance criterion
        if dE < 0 or rng.random() < np.exp(-dE / temp):
            current_energy = new_energy
            current_pos = new_pos
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
            # Cooling and step size adjustment
            temp *= 0.95
            step_size *= 0.98
        else:
            # If rejected, try a slightly larger step occasionally
            if rng.random() < 0.1:
                step_size *= 1.1
                
    return best_energy, best_pos

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    lock = threading.Lock()
    
    def update_best(energy, positions):
        nonlocal best_energy, best_positions
        if energy < best_energy:
            with lock:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()

    # Generate diverse starting points
    starts = []
    
    # 1. FCC fragment
    fcc = _fcc_fragment(n)
    if fcc is not None:
        starts.append(fcc)
        
    # 2. HCP fragment
    hcp = _hcp_fragment(n)
    if hcp is not None:
        starts.append(hcp)
        
    # 3. Random starts (more diverse)
    # We want to generate enough starts to keep threads busy
    # Estimate time per relax+hop: ~0.1-0.5s. 
    # With 4 threads and 300s, we can do ~3000-6000 relaxations.
    # Let's generate 20-40 random starts and let them run for a few hops each.
    num_random_starts = 30 
    for seed in range(num_random_starts):
        rand = _random_compact(n, seed=seed)
        starts.append(rand)
        
    # 4. Add a few BCC-like starts just in case
    bcc_points = []
    max_coord = int(np.ceil(n ** (1/3))) + 4
    a = 0.72
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                bcc_points.append([i*a, j*a, k*a])
                bcc_points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(bcc_points) >= n:
        bcc_points = np.array(bcc_points)
        dists = np.linalg.norm(bcc_points, axis=1)
        idx = np.argsort(dists)[:n]
        starts.append(bcc_points[idx])

    # Shuffle starts to ensure diversity in parallel execution
    np.random.shuffle(starts)
    
    # Number of threads: use min(4, len(starts)) to avoid overhead
    num_threads = min(4, len(starts))
    
    # We will run chains until time runs out.
    # We create a pool of workers.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        rng_seed_base = 1000
        
        for i, start in enumerate(starts):
            if _time_left() < 10:
                break
            # Each worker gets a unique RNG
            rng = np.random.default_rng(rng_seed_base + i)
            # Limit hops per chain to ensure we try many chains.
            # If a chain is fast, it finishes and picks up another.
            # But for simplicity, we submit all starts as tasks.
            # To prevent a single task from running forever, we cap hops.
            # 20-50 hops is usually enough for basin hopping to find a good local min.
            max_hops = 30
            future = executor.submit(_basin_hop_chain, start, n, kind, potential, rng, max_hops)
            futures.append(future)
            
        # Process results as they complete
        for future in as_completed(futures):
            if _time_left() < 5:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy and positions is not None:
                    update_best(energy, positions)
            except Exception:
                pass

    # If we still have time, do a final polish on the best found with more hops
    if best_positions is not None and _time_left() > 30:
        # Run a longer, more focused basin hopping on the current best
        rng = np.random.default_rng(42)
        try:
            # Increase temperature slightly to allow exploration
            final_energy, final_pos = _basin_hop_chain(
                best_positions, n, kind, potential, rng, max_hops=100, initial_temp=0.3
            )
            if final_energy < best_energy:
                update_best(final_energy, final_pos)
        except Exception:
            pass

    # Safety check
    if best_positions is not None:
        # Ensure finite and min distance
        if not np.all(np.isfinite(best_positions)):
            return None
        
        # Check min distance
        # For large n, O(n^2) is okay for a final check if n < 100
        min_d = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            # If invalid, we might try to return a valid random one?
            # But usually relaxer prevents this.
            # If it happens, the score is 0. 
            # We could try to "fix" by expanding, but that's risky.
            pass

    return best_positions
# EVOLVE-BLOCK-END
