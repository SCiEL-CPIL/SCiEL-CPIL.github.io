"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _makespan_from_c(processing_times, order, c_start):
    """Compute makespan starting from a given completion time vector."""
    if len(order) == 0:
        return c_start[-1] if len(c_start) > 0 else 0.0
    n_machines = processing_times.shape[1]
    c = c_start.copy()
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 148  # 2s safety margin
    
    # --- Phase 1: NEH Constructive Heuristic ---
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    current_order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_order) + 1):
            new_order = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order.insert(best_pos, job)
        
        if time.time() > deadline:
            break
    
    best_order = list(current_order)
    best_ms = _makespan(processing_times, best_order)
    
    # --- Phase 2: Local Search (Full Neighborhood Hill Climbing) ---
    # Repeat full passes until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best_order)
        for i in range(n):
            if time.time() > deadline:
                break
            job = best_order[i]
            temp_order = best_order[:i] + best_order[i+1:]
            
            best_pos_local = i
            best_ms_local = float('inf')
            
            for pos in range(len(temp_order) + 1):
                if pos == i:
                    continue
                new_order = temp_order[:pos] + [job] + temp_order[pos:]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms_local:
                    best_ms_local = ms
                    best_pos_local = pos
            
            if best_ms_local < best_ms:
                best_order = temp_order[:best_pos_local] + [job] + temp_order[best_pos_local:]
                best_ms = best_ms_local
                improved = True
                break  # Restart scan after improvement
    
    # --- Phase 3: Simulated Annealing ---
    if time.time() < deadline:
        current = list(best_order)
        current_ms = best_ms
        T = max(1.0, best_ms * 0.01)  # Scale temperature with problem size
        cooling = 0.999
        min_T = 1e-4
        
        while T > min_T and time.time() < deadline:
            n = len(current)
            if n < 2:
                break
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                T *= cooling
                continue
            
            # Use insertion move for larger neighborhood
            job = current[i]
            temp = current[:i] + current[i+1:]
            if i < j:
                j -= 1
            new_order = temp[:j] + [job] + temp[j:]
            
            new_ms = _makespan(processing_times, new_order)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current = new_order
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current)
            
            T *= cooling
    
    # --- Phase 4: Final Polish (Tight Local Search) ---
    # Do a few more passes of full neighborhood search with tighter time check
    if time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            improved = False
            n = len(best_order)
            for i in range(n):
                if time.time() > deadline:
                    break
                job = best_order[i]
                temp_order = best_order[:i] + best_order[i+1:]
                
                best_pos_local = i
                best_ms_local = float('inf')
                
                for pos in range(len(temp_order) + 1):
                    if pos == i:
                        continue
                    new_order = temp_order[:pos] + [job] + temp_order[pos:]
                    ms = _makespan(processing_times, new_order)
                    if ms < best_ms_local:
                        best_ms_local = ms
                        best_pos_local = pos
                
                if best_ms_local < best_ms:
                    best_order = temp_order[:best_pos_local] + [job] + temp_order[best_pos_local:]
                    best_ms = best_ms_local
                    improved = True
                    break
            if not improved:
                break
    
    return best_order
# EVOLVE-BLOCK-END
