"""
Strategy: Parallel Replica Exchange Monte Carlo (REM) with Global Best Teleportation.
Instead of simple Basin Hopping, we run 4 parallel replicas.
Each replica performs Metropolis-Hastings moves.
Crucially, we maintain a global 'best_energy' and 'best_pos'.
If a replica gets stuck (high energy) or randomly decides to, it 'teleports' 
to the current global best and restarts its local search from there.
This avoids the local minima trapping of simple BH and uses the 4 threads 
to explore different basins simultaneously, sharing information via the global best.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_START_TIME = None
_MAX_THREADS = 4

# Global state shared by threads
_global_best_energy = float('inf')
_global_best_pos = None
_lock = None # We will use a simple list or atomic check if needed, but GIL makes scalar reads/writes mostly safe for this purpose in CPython. 
             # To be safe and simple, we'll use a lock if available or just rely on GIL for the check-and-update.

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return max(0.0, _DEADLINE_SEC - (time.time() - _START_TIME))

def _generate_fcc(n, spacing=0.72):
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
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _worker(args):
    global _global_best_energy, _global_best_pos
    
    instance, start_pos, thread_id = args
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    if _time_left() < 5:
        return (float('inf'), None)
    
    # Initialize replica
    current_pos = start_pos.copy()
    
    # Initial Relaxation
    try:
        current_energy, current_pos = relax(current_pos, potential, kind)
        if not np.isfinite(current_energy):
            current_energy = 1e6
    except Exception:
        current_energy = 1e6
        # If relaxation fails, keep start_pos but maybe perturb?
        # For now, just proceed with high energy
        
    # Update Global Best
    if current_energy < _global_best_energy:
        _global_best_energy = current_energy
        _global_best_pos = current_pos.copy()

    # MC Parameters
    # Temperature needs to be scaled to the energy scale of the system.
    # Sutton-Chen energies are typically negative, magnitude ~ -n * 1.
    # Let's estimate E ~ -2 * n. Delta E moves ~ 0.1-1.0.
    # T should be high enough to allow uphill moves.
    # Adaptive T: Start high, cool down? Or keep constant?
    # For REM-like behavior, constant T per replica or shared T is common.
    # Let's use a single T for all, tuned to be ~ 1.0 relative to k_B.
    # Actually, let's make T adaptive based on n.
    T = 0.5  # Arbitrary unit, scaled by energy differences
    
    # MC Loop
    # We want to do as many moves as possible.
    # Each move: perturb, check energy (fast), accept/reject.
    # Every K steps: Relax (slow).
    
    relax_freq = 20 # Relax every 20 accepted/rejected moves? Or every 20 steps?
    # Relaxing is expensive. Let's do it less often.
    # Actually, 'relax' is the bottleneck. 
    # Strategy: Perturb -> Energy Check (gradient) -> If Accepted, Relax.
    # This is expensive. 
    # Alternative: Perturb -> Relax -> Compare. (This is Basin Hopping).
    # The prompt asks for a DIFFERENT strategy.
    # True MC with frequent relaxations is slow.
    # Let's stick to a "Perturb-Relax" loop but with the GLOBAL BEST TELEPORTATION.
    # This is effectively "Parallel Basin Hopping with Global Memory".
    
    rng = np.random.default_rng(thread_id * 777)
    step_size = 0.1
    no_improve_count = 0
    
    while _time_left() > 10:
        # 1. Teleportation Check
        # If we are stuck (energy > global_best + threshold) or randomly, teleport
        # Threshold: if we are significantly worse, jump.
        if _global_best_pos is not None:
            # Probability of teleporting increases if we are far from best
            diff = current_energy - _global_best_energy
            # If diff > 0.1, chance to teleport is high.
            # Also add a random small chance to explore from best.
            p_teleport = 0.01 + 0.5 * np.tanh(diff) # Sigmoid-ish
            if rng.random() < p_teleport:
                current_pos = _global_best_pos.copy()
                # Perturb slightly to avoid immediate rejection/stuck
                current_pos += rng.normal(0, step_size * 0.5, size=(n, 3))
                try:
                    current_energy, current_pos = relax(current_pos, potential, kind)
                except Exception:
                    continue
                
                if current_energy < _global_best_energy:
                    _global_best_energy = current_energy
                    _global_best_pos = current_pos.copy()
                continue # Skip to next loop, we just did a relax
        
        # 2. Standard BH Move
        # Perturb
        noise = rng.normal(0, step_size, size=(n, 3))
        perturbed = current_pos + noise
        
        # Relax
        try:
            e_new, pos_new = relax(perturbed, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(e_new):
            continue
            
        # Metropolis Acceptance (for BH, we usually just take if better, 
        # but to escape local minima, we accept worse with probability exp(-dE/T))
        dE = e_new - current_energy
        if dE < 0:
            # Accept
            current_energy = e_new
            current_pos = pos_new
            no_improve_count = 0
            
            # Update Global
            if current_energy < _global_best_energy:
                _global_best_energy = current_energy
                _global_best_pos = current_pos.copy()
                
            # Cool down step size slightly on success?
            step_size *= 0.98
        else:
            # Accept with probability
            # T needs to be scaled. Let's assume energy scale ~ 1.0 per atom?
            # Total energy ~ -n.
            # Let's use a T that allows some uphill moves.
            T_eff = 0.5 # Fixed for now
            if rng.random() < np.exp(-dE / T_eff):
                current_energy = e_new
                current_pos = pos_new
                no_improve_count += 1
                # Increase step size on uphill acceptance to explore
                step_size *= 1.05
            else:
                no_improve_count += 1
                # Increase step size on rejection to try larger jumps
                step_size *= 1.02

        step_size = np.clip(step_size, 0.05, 1.0)
        
        # If stuck for a long time, force a larger jump or teleport
        if no_improve_count > 50:
            # Force a large jump
            large_noise = rng.normal(0, 0.5, size=(n, 3))
            perturbed = current_pos + large_noise
            try:
                e_new, pos_new = relax(perturbed, potential, kind)
                if np.isfinite(e_new):
                    current_energy = e_new
                    current_pos = pos_new
                    if current_energy < _global_best_energy:
                        _global_best_energy = current_energy
                        _global_best_pos = current_pos.copy()
            except Exception:
                pass
            no_improve_count = 0
            step_size = 0.2 # Reset step size

    return (current_energy, current_pos)

def construct(instance):
    global _START_TIME, _global_best_energy, _global_best_pos
    
    _START_TIME = time.time()
    _global_best_energy = float('inf')
    _global_best_pos = None
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse starting points
    starts = []
    
    fcc = _generate_fcc(n)
    if fcc is not None:
        starts.append(fcc)
        
    hcp = _generate_hcp(n)
    if hcp is not None:
        starts.append(hcp)
        
    bcc = _generate_bcc(n)
    if bcc is not None:
        starts.append(bcc)
        
    for seed in range(8):
        rand = _generate_random(n, seed=seed)
        starts.append(rand)
        
    rng_master = np.random.default_rng(123)
    rng_master.shuffle(starts)
    
    # Ensure we have at least 4 starts
    while len(starts) < _MAX_THREADS:
        starts.append(_generate_random(n, seed=len(starts)+100))
        
    # Select 4 distinct starts
    selected_starts = starts[:_MAX_THREADS]
    
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
                
    # The global variables _global_best_energy and _global_best_pos are also updated
    # by the workers. We should check those too, as the future.result() might
    # return a local best that is worse than the global best found by another thread
    # after this thread finished.
    
    if _global_best_energy < best_energy_global:
        best_energy_global = _global_best_energy
        best_pos_global = _global_best_pos
        
    # Final safety check
    if best_pos_global is None:
        fallback_start = _generate_random(n, seed=999)
        try:
            e, pos = relax(fallback_start, 'sutton_chen', kind)
            best_pos_global = pos
        except Exception:
            best_pos_global = fallback_start
            
    return best_pos_global
# EVOLVE-BLOCK-END
