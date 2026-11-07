"""
Strategy: Parallel Multi-Start Global Optimization.
Leverages 4 threads to simultaneously explore different structural basins.
Each thread generates a diverse set of initial configurations (FCC, HCP, BCC, Random)
and performs local relaxation and limited basin hopping. The best result across
all threads is returned.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_START_TIME = None
_MAX_THREADS = 4

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return max(0, _DEADLINE_SEC - (time.time() - _START_TIME))

def _generate_fcc(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_hcp(n, spacing=0.72):
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

def _generate_bcc(n, spacing=0.72):
    """Generate n atoms from a BCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_random(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to all existing points
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: place remaining atoms randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _worker(args):
    """
    Worker function for parallel execution.
    args: (instance, start_pos, thread_id)
    Returns: (energy, positions)
    """
    instance, start_pos, thread_id = args
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Check time immediately
    if _time_left() < 5:
        return (float('inf'), None)
    
    best_energy = float('inf')
    best_pos = None
    
    # 1. Initial Relaxation
    try:
        e, pos = relax(start_pos, potential, kind)
        if np.isfinite(e):
            best_energy = e
            best_pos = pos
    except Exception:
        pass
    
    # 2. Local Basin Hopping (limited steps to fit in budget)
    if best_pos is not None and _time_left() > 10:
        rng = np.random.default_rng(thread_id * 1000 + 42)
        step_size = 0.15
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        # Estimate max hops based on remaining time
        # Assume ~0.1s per hop (relax + perturb)
        max_hops = int((_time_left() - 5) / 0.1)
        max_hops = min(max_hops, 50)  # Cap to prevent runaway
        
        for _ in range(max_hops):
            if _time_left() < 5:
                break
            
            noise = rng.normal(0, step_size, size=(n, 3))
            perturbed = current_pos + noise
            
            try:
                e_new, pos_new = relax(perturbed, potential, kind)
                if np.isfinite(e_new):
                    if e_new < current_energy:
                        current_energy = e_new
                        current_pos = pos_new
                        if e_new < best_energy:
                            best_energy = e_new
                            best_pos = pos_new.copy()
                        step_size *= 0.95
                    else:
                        step_size *= 1.05
                    step_size = np.clip(step_size, 0.05, 1.0)
            except Exception:
                continue
                
    return (best_energy, best_pos)

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse starting points
    starts = []
    
    # Structural templates
    fcc = _generate_fcc(n)
    if fcc is not None:
        starts.append(fcc)
        
    hcp = _generate_hcp(n)
    if hcp is not None:
        starts.append(hcp)
        
    bcc = _generate_bcc(n)
    if bcc is not None:
        starts.append(bcc)
        
    # Random starts (diverse seeds)
    for seed in range(8):
        rand = _generate_random(n, seed=seed)
        starts.append(rand)
        
    # Shuffle starts to ensure diversity across threads
    rng_master = np.random.default_rng(123)
    rng_master.shuffle(starts)
    
    # Prepare tasks for threads
    # We have 4 threads. Distribute starts among them.
    # Each thread will take a subset of starts and process them sequentially,
    # or we can just assign one start per thread and let them do their own BH.
    # Given the high cost of BH, it's better to have each thread focus on 
    # a high-quality start and do deep BH, rather than shallow BH on many.
    # However, multi-start is key. Let's assign the first 4 distinct starts to threads.
    # If we have more starts, we can round-robin or just pick the best 4.
    # For now, pick 4 diverse starts (FCC, HCP, BCC, Random) if available.
    
    selected_starts = []
    if fcc is not None: selected_starts.append(fcc)
    if hcp is not None: selected_starts.append(hcp)
    if bcc is not None: selected_starts.append(bcc)
    if len(starts) > 0: selected_starts.append(starts[0]) # One random
    
    # Ensure we have 4 tasks
    while len(selected_starts) < _MAX_THREADS:
        selected_starts.append(_generate_random(n, seed=len(selected_starts)+100))
        
    tasks = []
    for i in range(_MAX_THREADS):
        tasks.append((instance, selected_starts[i], i))
        
    best_energy_global = float('inf')
    best_pos_global = None
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=_MAX_THREADS) as executor:
        futures = {executor.submit(_worker, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            if _time_left() < 2:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy_global:
                    best_energy_global = energy
                    best_pos_global = positions
            except Exception:
                pass
                
    # Final safety check
    if best_pos_global is None:
        # Fallback to a simple relaxation of a random start
        fallback_start = _generate_random(n, seed=999)
        try:
            e, pos = relax(fallback_start, 'sutton_chen', kind)
            best_pos_global = pos
        except Exception:
            best_pos_global = fallback_start
            
    return best_pos_global
# EVOLVE-BLOCK-END
