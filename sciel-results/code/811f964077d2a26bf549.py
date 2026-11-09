"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from copy import deepcopy


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    if len(perm) == 0:
        return 0.0
    n_machines = pt.shape[1]
    # completion_time[j] = completion time of job j on the last machine
    # We simulate the flow shop
    # C[m][j] = completion time of job j on machine m
    # C[m][j] = max(C[m-1][j], C[m][j-1]) + pt[j][m]
    # But we only need the final makespan, so we can track per-machine completion times
    # Actually, for a permutation, we can compute it efficiently
    # Let's use a 1D array for the current completion time on each machine
    # But the standard way:
    # C[0][j] = C[0][j-1] + pt[j][0]
    # C[m][j] = max(C[m-1][j], C[m][j-1]) + pt[j][m]
    # Makespan = C[n_machines-1][n_jobs-1]
    
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # C[m] = completion time of the last scheduled job on machine m
    # But we need C[m][j] for the current job j
    # Let's use a 2D approach but optimized:
    # Actually, we can use a 1D array if we process jobs in order
    # For job j, on machine m:
    #   C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    # where C[j-1][m] is the completion time of previous job on machine m
    # and C[j][m-1] is the completion time of current job on previous machine
    
    # Let prev_machine_completion[m] = completion time of previous job on machine m
    # Let cur_machine_completion = completion time of current job on machine m-1
    
    prev_machine_completion = np.zeros(n_machines)
    
    for j in perm:
        cur_machine_completion = 0.0
        for m in range(n_machines):
            cur_machine_completion = max(prev_machine_completion[m], cur_machine_completion) + pt[j, m]
            prev_machine_completion[m] = cur_machine_completion
    
    return cur_machine_completion


def _neh_initial(pt):
    """NEH heuristic for initial solution."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Insert jobs one by one
    perm = [order[0]]
    for idx in range(1, n_jobs):
        j = order[idx]
        # Find best position to insert j
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(perm) + 1):
            # Try inserting j at position pos
            trial = perm[:pos] + [j] + perm[pos:]
            mk = _makespan(trial, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        perm.insert(best_pos, j)
    
    return perm


def _local_search(perm, pt, max_iterations=500, time_limit=10.0):
    """2-opt local search with random restarts."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_mk = _makespan(best_perm, pt)
    current_perm = list(perm)
    current_mk = best_mk
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while iteration < max_iterations:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try all pairs of positions (2-opt)
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                # Swap or reverse? Let's do swap first
                trial = list(current_perm)
                trial[i], trial[j] = trial[j], trial[i]
                mk = _makespan(trial, pt)
                if mk < current_mk - 1e-9:
                    current_perm = trial
                    current_mk = mk
                    improved = True
                    if current_mk < best_mk - 1e-9:
                        best_perm = list(current_perm)
                        best_mk = current_mk
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        iteration += 1
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 140s out of 150s
    
    pt = np.asarray(processing_times)
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Start with NEH heuristic
    perm = _neh_initial(pt)
    best_mk = _makespan(perm, pt)
    best_perm = list(perm)
    
    # Local search from NEH
    ls_result = _local_search(perm, pt, max_iterations=1000, time_limit=5.0)
    ls_mk = _makespan(ls_result, pt)
    if ls_mk < best_mk:
        best_perm = ls_result
        best_mk = ls_mk
    
    # Check time
    if time.time() > deadline - 20.0:
        return best_perm
    
    # Multi-start local search with random perturbations
    rng = np.random.default_rng(42)
    
    # Try several random restarts
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the best solution
        perturbed = list(best_perm)
        # Random swap
        for _ in range(3):
            i = rng.integers(0, n_jobs)
            j = rng.integers(0, n_jobs)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        # Local search from perturbed
        ls_result = _local_search(perturbed, pt, max_iterations=300, time_limit=3.0)
        ls_mk = _makespan(ls_result, pt)
        if ls_mk < best_mk:
            best_perm = ls_result
            best_mk = ls_mk
    
    # Final check
    if time.time() > deadline - 5.0:
        return best_perm
    
    # Try a few more NEH-like variations
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        
        # Random NEH: shuffle the order, then use NEH insertion
        order = list(range(n_jobs))
        rng.shuffle(order)
        
        perm2 = [order[0]]
        for idx in range(1, n_jobs):
            j = order[idx]
            best_pos = 0
            best_mk2 = float('inf')
            for pos in range(len(perm2) + 1):
                trial = perm2[:pos] + [j] + perm2[pos:]
                mk = _makespan(trial, pt)
                if mk < best_mk2:
                    best_mk2 = mk
                    best_pos = pos
            perm2.insert(best_pos, j)
        
        # Local search
        ls_result = _local_search(perm2, pt, max_iterations=200, time_limit=2.0)
        ls_mk = _makespan(ls_result, pt)
        if ls_mk < best_mk:
            best_perm = ls_result
            best_mk = ls_mk
    
    return best_perm
# EVOLVE-BLOCK-END
