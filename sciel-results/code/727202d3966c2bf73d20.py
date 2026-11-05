# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, P):
    """Compute makespan for a given permutation."""
    n_machines = P.shape[1]
    if not perm:
        return 0.0
    prev = np.zeros(n_machines)
    for job in perm:
        curr = np.empty(n_machines)
        curr[0] = prev[0] + P[job, 0]
        for m in range(1, n_machines):
            curr[m] = max(prev[m], curr[m-1]) + P[job, m]
        prev = curr
    return prev[-1]

def _neh_order(P, random_seed=None):
    """
    NEH heuristic with optional randomization.
    """
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    
    totals = P.sum(axis=1)
    if random_seed is not None:
        # Add small noise to break ties or slightly perturb order
        noise = np.random.RandomState(random_seed).uniform(0, 1, size=n_jobs) * 0.1 * totals.max()
        totals = totals + noise
    
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_mk = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            mk = _makespan(new_perm, P)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm

def _tabu_search(perm, P, time_limit=10.0, tabu_tenure=10):
    """
    Tabu Search with insertion moves.
    """
    n_jobs = len(perm)
    n_machines = P.shape[1]
    current_perm = list(perm)
    current_mk = _makespan(current_perm, P)
    best_perm = list(current_perm)
    best_mk = current_mk
    
    start_time = time.time()
    
    tabu_list = {} # (i, j) -> iteration when it expires
    iteration = 0
    
    while time.time() - start_time < time_limit:
        best_non_tabu_mk = float('inf')
        best_non_tabu_move = None
        best_non_tabu_perm = None
        
        best_aspiration_mk = float('inf')
        best_aspiration_move = None
        best_aspiration_perm = None
        
        # Generate neighbors
        for i in range(n_jobs):
            job = current_perm[i]
            perm_wo_job = current_perm[:i] + current_perm[i+1:]
            
            for j in range(n_jobs - 1):
                if j == i:
                    continue
                
                new_perm = perm_wo_job[:j] + [job] + perm_wo_job[j:]
                mk = _makespan(new_perm, P)
                
                move_key = (i, j)
                is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
                
                # Check aspiration
                if mk < best_mk and mk < best_aspiration_mk:
                    best_aspiration_mk = mk
                    best_aspiration_move = move_key
                    best_aspiration_perm = new_perm
                    
                # Check non-tabu
                if not is_tabu and mk < best_non_tabu_mk:
                    best_non_tabu_mk = mk
                    best_non_tabu_move = move_key
                    best_non_tabu_perm = new_perm
        
        # Select move
        selected_move = None
        selected_perm = None
        selected_mk = float('inf')
        
        # Prefer aspiration if it's better than best non-tabu (or if non-tabu is inf)
        if best_aspiration_move is not None:
            if best_aspiration_mk < best_non_tabu_mk or best_non_tabu_move is None:
                selected_move = best_aspiration_move
                selected_perm = best_aspiration_perm
                selected_mk = best_aspiration_mk
        
        if selected_move is None and best_non_tabu_move is not None:
            selected_move = best_non_tabu_move
            selected_perm = best_non_tabu_perm
            selected_mk = best_non_tabu_mk
            
        if selected_move is None:
            break # No valid moves
            
        i, j = selected_move
        current_perm = selected_perm
        current_mk = selected_mk
        
        if current_mk < best_mk:
            best_mk = current_mk
            best_perm = list(current_perm)
            
        # Update Tabu
        tabu_list[selected_move] = iteration + tabu_tenure
        iteration += 1
        
        # Clean up old tabu entries to save memory (optional, list is small)
        if iteration > 1000:
            # Remove entries that expired
            keys_to_remove = [k for k, v in tabu_list.items() if v <= iteration]
            for k in keys_to_remove:
                del tabu_list[k]
                
    return best_perm, best_mk

def order_jobs(processing_times):
    P = np.asarray(processing_times)
    n_jobs = P.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    total_time_limit = 110.0
    start_global = time.time()
    
    # Multi-start NEH
    best_perm = None
    best_mk = float('inf')
    
    # Run a few NEH variants
    num_starts = 5
    time_per_start = (total_time_limit * 0.2) / num_starts # 20% time for initial solutions
    
    for k in range(num_starts):
        if time.time() - start_global > total_time_limit * 0.2:
            break
        perm = _neh_order(P, random_seed=k)
        mk = _makespan(perm, P)
        if mk < best_mk:
            best_mk = mk
            best_perm = perm
            
    # If no start succeeded (shouldn't happen), fallback to standard NEH
    if best_perm is None:
        best_perm = _neh_order(P)
        best_mk = _makespan(best_perm, P)
        
    # Tabu Search
    ts_time_limit = total_time_limit * 0.7
    best_perm, best_mk = _tabu_search(best_perm, P, time_limit=ts_time_limit)
    
    # Polish
    elapsed = time.time() - start_global
    remaining = total_time_limit - elapsed - 2.0
    if remaining > 1.0:
        # Best improvement polish
        improved = True
        while improved and time.time() - start_global < total_time_limit - 2.0:
            improved = False
            n_jobs = len(best_perm)
            for i in range(n_jobs):
                job = best_perm[i]
                perm_wo = best_perm[:i] + best_perm[i+1:]
                for j in range(n_jobs - 1):
                    if j == i: continue
                    new_perm = perm_wo[:j] + [job] + perm_wo[j:]
                    mk = _makespan(new_perm, P)
                    if mk < best_mk:
                        best_perm = new_perm
                        best_mk = mk
                        improved = True
                        break
                if improved:
                    break
                    
    return list(best_perm)

# EVOLVE-BLOCK-END
