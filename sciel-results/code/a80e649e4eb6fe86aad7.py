"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation efficiently."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    return machine_times[-1]


def _makespan_delta(perm, processing_times, i, j):
    """
    Compute the makespan after swapping positions i and j in perm.
    Only recompute the suffix from min(i,j) onward.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1 or i == j:
        return _makespan(perm, processing_times)
    
    start = min(i, j)
    
    # Compute completion times up to position start-1
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for k in range(start):
        job = perm[k]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    # Process from start to end with the swapped positions
    for k in range(start, n_jobs):
        if k == i:
            job = perm[j]
        elif k == j:
            job = perm[i]
        else:
            job = perm[k]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    return machine_times[-1]


def _neh_order(processing_times, seed=None):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling with optional randomization."""
    n_jobs, n_machines = processing_times.shape
    rng = random.Random(seed)
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # For larger instances, add some randomization to the order
    if seed is not None and n_jobs > 10:
        # Slightly shuffle the sorted order
        for _ in range(min(5, n_jobs // 10)):
            idx = rng.randint(0, n_jobs - 2)
            sorted_jobs[idx], sorted_jobs[idx+1] = sorted_jobs[idx+1], sorted_jobs[idx]
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        # Use full makespan for NEH insertion to ensure correctness and simplicity
        # NEH is O(N^2 * M) which is acceptable for initialization
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _random_order(processing_times, seed=None):
    """Generate a random permutation."""
    n_jobs = processing_times.shape[0]
    perm = list(range(n_jobs))
    rng = random.Random(seed)
    rng.shuffle(perm)
    return perm


def _improve_order(perm, processing_times, time_limit=10.0, seed=None):
    """
    Improve permutation using a combination of 2-opt swaps and simulated annealing.
    Uses _makespan_delta for fast swap evaluation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    rng = random.Random(seed)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    start_time = time.time()
    
    # Phase 1: Local search with adjacent swaps
    # Using delta evaluation for speed
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.3:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit * 0.3:
                break
            # Try swapping adjacent positions i and i+1
            candidate_ms = _makespan_delta(current_perm, processing_times, i, i + 1)
            if candidate_ms < best_ms - 1e-9:
                # Apply the swap to current_perm
                current_perm[i], current_perm[i + 1] = current_perm[i + 1], current_perm[i]
                best_ms = candidate_ms
                best_perm = list(current_perm)
                current_ms = candidate_ms
                improved = True
    
    # Phase 2: Simulated annealing with random swaps and inserts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        t0 = max(10.0, best_ms * 0.01)
        t_end = 0.1
        cooling_rate = 0.99
        
        temperature = t0
        while temperature > t_end and (time.time() - start_time) < time_limit:
            move_type = rng.random()
            
            if move_type < 0.8:
                # Swap move (use fast delta evaluation)
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j:
                    continue
                
                candidate_ms = _makespan_delta(current_perm, processing_times, i, j)
                
                # Check if we should accept
                delta = candidate_ms - current_ms
                if delta < 0 or rng.random() < np.exp(-delta / temperature):
                    # Apply swap
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    current_ms = candidate_ms
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best_perm = list(current_perm)
            else:
                # Insert move (use full makespan, slower but necessary for diversity)
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j:
                    continue
                
                candidate = list(current_perm)
                job = candidate.pop(i)
                candidate.insert(j, job)
                candidate_ms = _makespan(candidate, processing_times)
                
                delta = candidate_ms - current_ms
                if delta < 0 or rng.random() < np.exp(-delta / temperature):
                    current_perm = candidate
                    current_ms = candidate_ms
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best_perm = list(current_perm)
            
            temperature *= cooling_rate
    
    # Phase 3: Final polish with tight tolerance (adjacent swaps)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 0.5:
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(n_jobs - 1):
                if (time.time() - start_time) >= time_limit:
                    break
                candidate_ms = _makespan_delta(current_perm, processing_times, i, i + 1)
                if candidate_ms < best_ms - 1e-12:
                    current_perm[i], current_perm[i + 1] = current_perm[i + 1], current_perm[i]
                    best_ms = candidate_ms
                    best_perm = list(current_perm)
                    current_ms = candidate_ms
                    improved = True
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses multiple NEH variants, random starts, and local search + SA for improvement.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (150s limit, use 145s)
    global_deadline = time.time() + 145.0
    
    # Generate multiple candidate permutations
    candidates = []
    
    # 1. Standard NEH
    neh_perm = _neh_order(processing_times, seed=None)
    candidates.append(neh_perm)
    
    # 2. NEH with slight randomization
    if n_jobs > 5:
        neh_rand_perm = _neh_order(processing_times, seed=42)
        if neh_rand_perm != neh_perm:
            candidates.append(neh_rand_perm)
    
    # 3. Random permutation
    rand_perm = _random_order(processing_times, seed=123)
    candidates.append(rand_perm)
    
    # 4. SPT (Shortest Processing Time) order
    totals = processing_times.sum(axis=1)
    spt_perm = list(np.argsort(totals))
    candidates.append(spt_perm)
    
    # Evaluate all candidates and pick the best
    best_perm = candidates[0]
    best_ms = _makespan(best_perm, processing_times)
    for cand in candidates[1:]:
        ms = _makespan(cand, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand
    
    # Improve with local search + SA
    elapsed = time.time() - (global_deadline - 145.0)
    remaining = global_deadline - time.time()
    
    if remaining > 2.0:
        # Use most of the remaining time for improvement
        improve_time = min(remaining - 1.0, 135.0)
        improved_perm, improved_ms = _improve_order(
            best_perm, processing_times, time_limit=improve_time, seed=42
        )
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # If we still have time, try another round of improvement from a different start
    remaining = global_deadline - time.time()
    if remaining > 5.0 and n_jobs > 5:
        # Try improving from a random start
        rand_perm2 = _random_order(processing_times, seed=999)
        rand_ms = _makespan(rand_perm2, processing_times)
        
        improve_time2 = min(remaining - 1.0, 60.0)
        improved_perm2, improved_ms2 = _improve_order(
            rand_perm2, processing_times, time_limit=improve_time2, seed=777
        )
        if improved_ms2 < best_ms:
            best_perm = improved_perm2
            best_ms = improved_ms2
    
    return best_perm
# EVOLVE-BLOCK-END
