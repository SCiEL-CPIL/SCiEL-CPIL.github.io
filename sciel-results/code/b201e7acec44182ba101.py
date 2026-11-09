import numpy as np
import time

def _makespan(perm, processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    prev_row = np.zeros(n_machines)
    curr_row = np.zeros(n_machines)
    for job in perm:
        p = processing_times[job]
        curr_row[0] = prev_row[0] + p[0]
        for m in range(1, n_machines):
            curr_row[m] = max(curr_row[m-1], prev_row[m]) + p[m]
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    current_perm = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_makespan = np.inf
        for pos in range(len(current_perm) + 1):
            candidate = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
    
    improved = True
    passes = 0
    max_passes = 100
    t_start = time.time()
    
    while improved and passes < max_passes and (time.time() - t_start) < 100:
        improved = False
        passes += 1
        n = len(current_perm)
        current_ms = _makespan(current_perm, processing_times)
        
        for i in range(n):
            job_to_move = current_perm[i]
            cand_base = current_perm[:i] + current_perm[i+1:]
            
            for insert_pos in range(n):
                if insert_pos == i:
                    continue
                candidate = cand_base[:insert_pos] + [job_to_move] + cand_base[insert_pos:]
                ms = _makespan(candidate, processing_times)
                if ms < current_ms:
                    current_perm = candidate
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
                
    return current_perm
