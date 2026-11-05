"""Rewritten program for flow-shop permutation task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(pt, perm):
    """Compute makespan for a given permutation using DP."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0
    
    # Use pure Python lists for speed on small M
    # prev[m] = completion time of previous job on machine m
    prev = [0.0] * n_machines
    for j in perm:
        # curr[m] = completion time of job j on machine m
        curr = [0.0] * n_machines
        curr[0] = prev[0] + pt[j, 0]
        for m in range(1, n_machines):
            # max(prev[m], curr[m-1]) + pt[j, m]
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt[j, m]
        prev = curr
    return prev[-1]

def _neh_initial(pt):
    """NEH heuristic: sort by total processing time descending, insert each job at best position."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm

def _palmer_initial(pt):
    """Palmer's heuristic: sort by (n-1)*sum_first_half - sum_second_half."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))

def _lpt_initial(pt):
    """LPT: sort by total processing time descending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))

def _spt_initial(pt):
    """SPT: sort by total processing time ascending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))

def _random_initial(n_jobs, rng):
    """Random permutation."""
    return list(rng.permutation(n_jobs))

def _local_search(pt, perm, deadline, max_evals=50000):
    """
    Local search with adjacent swap neighborhood.
    Returns best permutation and makespan found.
    """
    n_jobs = len(perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    eval_count = 0
    
    while eval_count < max_evals:
        if time.time() > deadline:
            break
        
        improved = False
        # Try all adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() > deadline or eval_count >= max_evals:
                break
            
            # Swap i and i+1
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            new_ms = _makespan(pt, current_perm)
            eval_count += 1
            
            if new_ms < current_ms:
                current_ms = new_ms
                improved = True
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
            else:
                # Swap back
                current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
        
        if not improved:
            break
    
    return best_perm, best_ms

def _simulated_annealing(pt, perm, deadline, max_iter=10000):
    """
    Simulated annealing with adjacent swap neighborhood.
    """
    n_jobs = len(perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.1, 1.0)
    T_min = max(current_ms * 0.001, 0.1)
    cooling = 0.99
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: pick random adjacent pair to swap
        i = random.randint(0, n_jobs - 2)
        
        # Swap
        current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
        new_ms = _makespan(pt, current_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        else:
            # Swap back
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multi-start local search
    with NEH, Palmer, and other heuristics, plus simulated annealing.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard: 145s to leave 5s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 145
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH (best single heuristic)
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer's heuristic
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 6. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 7. Sort by mean descending
    candidates.append(list(np.argsort(-pt.mean(axis=1))))
    
    # 8-11. Random permutations (reduced to save time for search)
    for _ in range(4):
        candidates.append(_random_initial(n_jobs, rng))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Run local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 10:  # Leave 10s for final SA
            break
        
        perm, ms = _local_search(pt, cand, deadline - 10)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Run simulated annealing from best solution found
    if time.time() < deadline - 5:
        perm, ms = _simulated_annealing(pt, best_perm, deadline - 5, max_iter=20000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we still have time, run SA from a few other good starts
    if time.time() < deadline - 2:
        for cand in candidates[:5]:
            if time.time() > deadline - 2:
                break
            perm, ms = _simulated_annealing(pt, cand, deadline - 2, max_iter=5000)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm
    
    return best_perm

# EVOLVE-BLOCK-END
