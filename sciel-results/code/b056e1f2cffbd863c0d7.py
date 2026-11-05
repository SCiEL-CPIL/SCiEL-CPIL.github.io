"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(pt, perm):
    """Compute makespan for a given permutation of jobs."""
    if len(perm) == 0:
        return 0
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # C[m] is the completion time of the last job on machine m
    C = np.zeros(n_machines)
    makespan = 0
    
    for job in perm:
        for m in range(n_machines):
            # Job starts after previous job on this machine finishes (C[m])
            # and after previous job on this sequence finishes (C[m-1] if m>0 else 0)
            if m == 0:
                start = C[0]
            else:
                start = max(C[m], C[m-1])
            
            finish = start + pt[job, m]
            C[m] = finish
            if m == n_machines - 1:
                makespan = finish
                
    return makespan

def _try_2opt(pt, perm, best_perm, best_ms):
    """Try to improve permutation via 2-opt moves. Returns (best_perm, best_ms, improved)."""
    n = len(perm)
    if n < 2:
        return perm, best_ms, False
    
    improved = False
    # Iterate through possible swaps
    # For efficiency, we can limit the number of passes or just do one full pass of swaps
    # A simple 2-opt is swapping i and j where i < j
    # To keep it fast, we'll do a few passes of random swaps or structured swaps.
    # Let's do a structured pass: try swapping i with i+1 (adjacent) and i with i+2, etc.
    # Actually, standard 2-opt in TSP is reversing a segment. In flow shop, swapping two jobs is a common move.
    # Let's try all pairs (i, j) with i < j. Complexity O(n^2 * M).
    # If n is large (e.g., 100), n^2 = 10,000. M=10. 100,000 ops. Fast.
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            ms = _makespan(pt, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_perm = new_perm
                improved = True
    return best_perm, best_ms, improved

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate initial candidates
    candidates = []
    
    # 1. LPT (Longest Processing Time)
    candidates.append(np.argsort(-pt.sum(axis=1)))
    
    # 2. SPT (Shortest Processing Time)
    candidates.append(np.argsort(pt.sum(axis=1)))
    
    # 3. Random
    for _ in range(5):
        perm = np.random.permutation(n_jobs)
        candidates.append(perm)
    
    # 4. NEH-like (constructive)
    # Sort by total processing time descending
    order = np.argsort(-pt.sum(axis=1))
    neh_perm = []
    for job in order:
        # Find best position to insert job
        best_pos = -1
        best_ms = np.inf
        for pos in range(len(neh_perm) + 1):
            test_perm = neh_perm[:pos] + [job] + neh_perm[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        neh_perm.insert(best_pos, job)
    candidates.append(np.array(neh_perm))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Update best with other candidates
    for perm in candidates[1:]:
        ms = _makespan(pt, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Local Search Loop
    # Use 2-opt swaps
    # We need to be careful with time.
    # Estimate cost of one full 2-opt pass: n^2 * M ops.
    # If n=50, 2500 * 10 = 25,000 ops. Very fast.
    # If n=200, 40,000 * 10 = 400,000 ops. Still fast (<0.1s in Python if optimized, but loop overhead might make it ~1s).
    # Let's run as many passes as possible before deadline.
    
    while True:
        current_time = time.time()
        if current_time > deadline - 5.0:  # Stop 5s before hard deadline to be safe
            break
        
        improved = False
        # Perform one full pass of 2-opt swaps
        best_perm, best_ms, improved = _try_2opt(pt, best_perm, best_ms)
        
        if not improved:
            # No improvement in full pass, try random restarts or break
            # If we've converged, maybe try a few random perturbations?
            # For now, break to save time, or try a random swap to escape local opt.
            # Let's try a few random swaps to escape local optima if time permits.
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i == j:
                    continue
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _makespan(pt, new_perm)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
            if not improved:
                break
        else:
            # Continue searching
            pass

    return list(best_perm)
# EVOLVE-BLOCK-END
