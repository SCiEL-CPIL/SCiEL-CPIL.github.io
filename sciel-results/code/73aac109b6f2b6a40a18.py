import numpy as np
import time

def _makespan(perm, processing_times):
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0
    
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    
    for j in range(n):
        job_idx = perm[j]
        p = processing_times[job_idx]
        
        if j == 0:
            curr_row[0] = p[0]
            for k in range(1, m):
                curr_row[k] = curr_row[k-1] + p[k]
        else:
            curr_row[0] = prev_row[0] + p[0]
            for k in range(1, m):
                curr_row[k] = max(curr_row[k-1], prev_row[k]) + p[k]
        
        prev_row, curr_row = curr_row, prev_row
        
    return prev_row[-1]

def _neh_heuristic(processing_times):
    n = processing_times.shape[0]
    if n == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Start with the first job
    perm = [sorted_jobs[0]]
    
    # Insert the remaining jobs one by one in the best position
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            # Create a new permutation by inserting job at pos
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
        
    return perm

def order_jobs(processing_times):
    n = processing_times.shape[0]
    m = processing_times.shape[1]
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # 1. Initial NEH heuristic
    best_perm = _neh_heuristic(processing_times)
    best_makespan = _makespan(best_perm, processing_times)
    
    # 2. Local Search: 2-opt and Or-opt
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for safety
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Try swapping i with j
            for j in range(i+1, n):
                if time.time() - start_time > time_limit:
                    break
                # Swap i and j
                best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
                new_makespan = _makespan(best_perm, processing_times)
                
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    improved = True
                else:
                    # Swap back
                    best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
            
            # Try moving i to a different position
            for j in range(n):
                if time.time() - start_time > time_limit:
                    break
                if i == j:
                    continue
                # Move i to j
                job = best_perm.pop(i)
                if j > i:
                    j -= 1
                best_perm.insert(j, job)
                
                new_makespan = _makespan(best_perm, processing_times)
                
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    improved = True
                else:
                    # Move back
                    job = best_perm.pop(j)
                    if i > j:
                        i -= 1
                    best_perm.insert(i, job)
        
        if time.time() - start_time > time_limit:
            break
            
    return best_perm
