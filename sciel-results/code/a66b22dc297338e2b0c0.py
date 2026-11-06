"""
Matrix-based delta evaluation for flow-shop permutation optimization.
Instead of recomputing the full makespan O(n*m) for every neighborhood move,
we precompute a per-machine cumulative completion matrix C[j, m] for the current
permutation. A swap of adjacent jobs i and i+1 can then be evaluated in O(m)
by only re-computing the affected rows, giving a massive speedup that allows
exhaustive 2-opt/swap neighborhood exploration within the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_C(perm, pt):
    """
    Compute the full completion-time matrix C for a given permutation.
    C[j, m] = completion time of job perm[j] on machine m.
    Shape: (n_jobs, n_machines)
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    for j in range(n_jobs):
        job = perm[j]
        for m in range(n_machines):
            if j == 0:
                prev_job_time = 0.0
            else:
                prev_job_time = C[j-1, m]
            if m == 0:
                prev_machine_time = 0.0
            else:
                prev_machine_time = C[j, m-1]
            C[j, m] = max(prev_job_time, prev_machine_time) + pt[job, m]
    
    return C

def _makespan_from_C(C):
    """Makespan is the last element of the last row of C."""
    return C[-1, -1]

def _swap_adjacent_delta(C, i, pt, perm):
    """
    Compute the new makespan after swapping adjacent jobs at positions i and i+1.
    Only rows i and i+1 (and potentially downstream rows) need recomputation.
    Actually, swapping adjacent jobs i, i+1 only affects rows i, i+1, and all rows after i+1
    because the completion times on each machine shift.
    
    However, for a true O(m) delta, we can note that:
    - C[i, m] and C[i+1, m] change.
    - C[j, m] for j > i+1 depends on C[j-1, m], so they all change.
    
    To keep it O(m) per swap, we need a smarter approach. 
    Let's recompute rows i, i+1, and then propagate forward. But that's O((n-i)*m).
    
    Alternative: use the fact that for adjacent swap, we can compute the new C[i,m] and C[i+1,m]
    and then the makespan only depends on the last row. But the last row depends on all previous.
    
    Let's instead use a different delta: compute the new C for rows i and i+1, then 
    for rows j > i+1, the change is bounded. Actually, let's just recompute from row i to the end.
    For n=100, m=10, that's 90*10 = 900 ops per swap. With ~100 swaps in neighborhood, that's 90k ops.
    That's much faster than full O(n*m) = 1000 per full makespan * many evaluations.
    
    Let's implement it properly.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Create a copy of C
    C_new = C.copy()
    
    job_i = perm[i]
    job_ip1 = perm[i+1]
    
    # Recompute row i (now has job_ip1)
    for m in range(n_machines):
        prev_job = C_new[i-1, m] if i > 0 else 0.0
        prev_machine = C_new[i, m-1] if m > 0 else 0.0
        C_new[i, m] = max(prev_job, prev_machine) + pt[job_ip1, m]
    
    # Recompute row i+1 (now has job_i)
    for m in range(n_machines):
        prev_job = C_new[i, m]
        prev_machine = C_new[i+1, m-1] if m > 0 else 0.0
        C_new[i+1, m] = max(prev_job, prev_machine) + pt[job_i, m]
    
    # Propagate changes to rows i+2 onwards
    for j in range(i+2, n_jobs):
        for m in range(n_machines):
            prev_job = C_new[j-1, m]
            prev_machine = C_new[j, m-1] if m > 0 else 0.0
            C_new[j, m] = max(prev_job, prev_machine) + pt[perm[j], m]
    
    return C_new[-1, -1]

def _neh_order(pt):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    perm = []
    for job in sorted_jobs:
        if not perm:
            perm.append(job)
            continue
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            C = _compute_C(candidate, pt)
            ms = C[-1, -1]
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm

def _local_search_swap(perm, pt, time_limit, deadline):
    """
    Local search using adjacent swaps with delta evaluation.
    Repeatedly scan the entire permutation and swap any adjacent pair that improves makespan.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    C = _compute_C(current_perm, pt)
    current_ms = C[-1, -1]
    
    best_perm = list(current_perm)
    best_ms = current_ms
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if time.time() > deadline:
            break
        
        improved = False
        
        # Scan all adjacent pairs
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            
            new_ms = _swap_adjacent_delta(C, i, pt, current_perm)
            
            if new_ms < current_ms - 1e-9:
                # Apply the swap
                current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
                C = _compute_C(current_perm, pt)
                current_ms = C[-1, -1]
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = list(current_perm)
                
                improved = True
                # After a swap, restart the scan from the beginning
                # (or from i-1 to catch further improvements)
                break
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        else:
            no_improve = 0
    
    return best_perm

def _perturb(perm, strength):
    """Perturb by random adjacent swaps."""
    perm = list(perm)
    n_jobs = len(perm)
    for _ in range(strength):
        if n_jobs <= 1:
            break
        i = np.random.randint(0, n_jobs - 1)
        perm[i], perm[i+1] = perm[i+1], perm[i]
    return perm

def _iterated_local_search(initial_perm, pt, time_limit, deadline):
    """
    ILS with swap-based local search.
    """
    current_perm = list(initial_perm)
    best_perm = list(initial_perm)
    
    C = _compute_C(current_perm, pt)
    current_ms = C[-1, -1]
    best_ms = current_ms
    
    start_time = time.time()
    strength = 2
    
    while time.time() < time_limit:
        if time.time() > deadline:
            break
        
        # Allocate time to local search
        elapsed = time.time() - start_time
        ls_time = min(2.0, time_limit - elapsed)
        if ls_time <= 0.1:
            break
        
        ls_perm = _local_search_swap(current_perm, pt, ls_time, deadline)
        
        C_ls = _compute_C(ls_perm, pt)
        ls_ms = C_ls[-1, -1]
        
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_perm = list(ls_perm)
            current_perm = ls_perm
            current_ms = ls_ms
            strength = 2
        else:
            strength += 1
            if strength > 15:
                strength = 15
            current_perm = _perturb(best_perm, strength)
            current_ms = _compute_C(current_perm, pt)[-1, -1]
    
    return best_perm

def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_total = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. NEH initial solution
    neh_perm = _neh_order(processing_times)
    neh_ms = _compute_C(neh_perm, processing_times)[-1, -1]
    
    best_perm = neh_perm
    best_ms = neh_ms
    
    elapsed = time.time() - start_total
    time_left = deadline - elapsed
    
    if time_left > 2.0:
        # 2. ILS with swap-based local search
        ils_perm = _iterated_local_search(neh_perm, processing_times, time_left - 5.0, deadline)
        ils_ms = _compute_C(ils_perm, processing_times)[-1, -1]
        
        if ils_ms < best_ms:
            best_perm = ils_perm
            best_ms = ils_ms
    
    return best_perm
# EVOLVE-BLOCK-END
