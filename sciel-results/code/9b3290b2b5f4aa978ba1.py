"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    if m == 0:
        return 0.0
    
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    # We only need the last row's last element.
    # prev_row[k] holds C[j-1, k]
    # curr_row[k] holds C[j, k]
    
    prev_row = np.zeros(m)
    for j in perm:
        curr_row = np.zeros(m)
        # Machine 0
        curr_row[0] = prev_row[0] + pt[j, 0]
        # Other machines
        for k in range(1, m):
            curr_row[k] = max(prev_row[k], curr_row[k-1]) + pt[j, k]
        prev_row = curr_row
        
    return prev_row[-1]

def _neh_heuristic(pt):
    """
    NEH heuristic for flow shop.
    Returns a permutation of job indices.
    """
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
        
    # 1. Sort jobs by total processing time (descending)
    totals = pt.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # 2. Insert jobs one by one
    current_perm = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_mk = float('inf')
        # Try inserting at each position
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(new_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _delta_swap(perm, pt, i, j):
    """
    Calculate the exact change in makespan if we swap jobs at positions i and j.
    This requires recalculating the makespan for the affected segment.
    However, a simple recalculation of the whole makespan is O(n*m).
    For small n, this is fast enough.
    For larger n, we can try to optimize, but let's start with full recalculation
    inside the loop if it's fast enough, or use a smarter delta.
    
    Actually, let's just return the new makespan and let the caller handle it,
    or implement a fast delta.
    
    A fast delta for swap is complex. Let's just recalculate makespan for the candidate.
    If n is small (e.g. < 50), O(n*m) is very fast (microseconds).
    """
    candidate = perm.copy()
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return _makespan(candidate, pt)

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # 1. Initial solution: NEH
    current_perm = _neh_heuristic(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Local Search / Simulated Annealing
    # We will use a hybrid: Intensive local search (swap/insert) + SA for diversification
    
    # Precompute some stats for temperature scaling
    total_p = processing_times.sum()
    T = total_p * 0.5
    cooling = 0.99
    min_T = 1e-5
    
    n_jobs = n
    
    # Helper to check time
    def _time_left():
        return time.time() < deadline
    
    # Main Loop
    while _time_left():
        if T < min_T:
            # Restart with perturbation if time allows
            if not _time_left():
                break
            # Perturb best solution
            current_perm = best_perm[:]
            # Random swap
            i, j = random.sample(range(n_jobs), 2)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_mk = _makespan(current_perm, processing_times)
            T = total_p * 0.1
        else:
            # Generate neighbor: Swap
            # For small n, swap is good. For larger, insertion might be better.
            # Let's stick to swap for simplicity and speed, or mix.
            i, j = random.sample(range(n_jobs), 2)
            candidate = current_perm.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            cand_mk = _makespan(candidate, processing_times)
            delta = cand_mk - current_mk
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_perm = candidate
                current_mk = cand_mk
                
                if current_mk < best_mk:
                    best_mk = current_mk
                    best_perm = current_perm[:]
            
            T *= cooling
            
            # Optional: Intensive Local Search around current best
            # If we find a new best, do a quick local search to polish
            if current_mk == best_mk and _time_left():
                # Try all single swaps to see if we can improve further
                improved = True
                while improved and _time_left():
                    improved = False
                    for i in range(n_jobs):
                        if not _time_left():
                            break
                        for j in range(i+1, n_jobs):
                            if not _time_left():
                                break
                            cand = current_perm.copy()
                            cand[i], cand[j] = cand[j], cand[i]
                            mk = _makespan(cand, processing_times)
                            if mk < current_mk:
                                current_perm = cand
                                current_mk = mk
                                best_mk = mk
                                best_perm = current_perm[:]
                                improved = True
                                break
                        if improved:
                            break
                
    return list(best_perm)
# EVOLVE-BLOCK-END
