"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_completion_matrix(perm, pt):
    """
    Compute the full completion time matrix C for a given permutation.
    C[i, j] is the completion time of job perm[i] on machine j.
    Returns a 2D numpy array of shape (n, m) and the makespan (C[-1, -1]).
    """
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return np.zeros((0, m)), 0.0
    
    C = np.zeros((n, m))
    # First job
    C[0, 0] = pt[perm[0], 0]
    for j in range(1, m):
        C[0, j] = C[0, j-1] + pt[perm[0], j]
        
    # Subsequent jobs
    for i in range(1, n):
        C[i, 0] = C[i-1, 0] + pt[perm[i], 0]
        for j in range(1, m):
            C[i, j] = max(C[i-1, j], C[i, j-1]) + pt[perm[i], j]
            
    return C, C[-1, -1]

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    _, mk = _compute_completion_matrix(perm, pt)
    return mk

def _delta_insert(perm, pt, i, j):
    """
    Calculate the change in makespan if we move job at position i to position j (insertion).
    This is done by recalculating the completion times for the affected segment.
    However, a full recalculation is O(n*m). 
    For a truly different strategy, we can use a greedy construction with lookahead 
    or a different metaheuristic like Ant Colony Optimization (ACO) or Tabu Search with a specific structure.
    
    Given the prompt asks for a DIFFERENT strategy, let's implement a **Greedy Randomized Adaptive Search Procedure (GRASP)** 
    combined with a **Variable Neighborhood Descent (VND)** local search.
    
    GRASP:
    1. Constructive Phase: Build a solution greedily but with randomization (RCL - Restricted Candidate List).
    2. Local Search Phase: Refine the solution using VND (swap and insert neighborhoods).
    3. Repeat until time limit.
    
    This is distinct from SA which uses temperature-based acceptance. GRASP is a multi-start heuristic.
    """
    pass

def _neh_heuristic(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic for initial solution.
    1. Sort jobs by total processing time in non-increasing order.
    2. Insert each job in the best position in the current partial sequence.
    """
    n, m = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    current_perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(new_perm, processing_times)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return np.array(current_perm)

def _local_search_vnd(perm, pt, deadline):
    """
    Variable Neighborhood Descent (VND) local search.
    Neighborhoods:
    N1: Swap adjacent jobs (2-opt style but swap)
    N2: Swap any two jobs
    N3: Insertion (move a job to any other position)
    
    We cycle through these neighborhoods. If a move improves, we restart from N1.
    """
    n = len(perm)
    m = pt.shape[1]
    current_perm = perm.copy()
    current_mk = _makespan(current_perm, pt)
    best_perm = current_perm.copy()
    best_mk = current_mk
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # N1: Swap adjacent
        for i in range(n - 1):
            if time.time() > deadline:
                return best_perm, best_mk
            candidate = current_perm.copy()
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            cand_mk = _makespan(candidate, pt)
            if cand_mk < current_mk:
                current_perm = candidate
                current_mk = cand_mk
                if current_mk < best_mk:
                    best_mk = current_mk
                    best_perm = current_perm.copy()
                improved = True
                break # Restart from beginning of N1
        
        if not improved:
            # N2: Swap any two
            for i in range(n):
                if time.time() > deadline:
                    return best_perm, best_mk
                for j in range(i + 1, n):
                    candidate = current_perm.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    cand_mk = _makespan(candidate, pt)
                    if cand_mk < current_mk:
                        current_perm = candidate
                        current_mk = cand_mk
                        if current_mk < best_mk:
                            best_mk = current_mk
                            best_perm = current_perm.copy()
                        improved = True
                        break
                if improved:
                    break # Restart from beginning of N2
        
        if not improved:
            # N3: Insertion
            for i in range(n):
                if time.time() > deadline:
                    return best_perm, best_mk
                    job = current_perm[i]
                    # Remove job
                    new_perm = np.delete(current_perm, i)
                    for j in range(len(new_perm) + 1):
                        if j == i:
                            continue
                        candidate = np.insert(new_perm, j, job)
                        cand_mk = _makespan(candidate, pt)
                        if cand_mk < current_mk:
                            current_perm = candidate
                            current_mk = cand_mk
                            if current_mk < best_mk:
                                best_mk = current_mk
                                best_perm = current_perm.copy()
                            improved = True
                            break
                    if improved:
                        break # Restart from beginning of N3
                        
    return best_perm, best_mk

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # GRASP: Multi-start heuristic
    best_perm_overall = None
    best_mk_overall = float('inf')
    
    # Number of iterations depends on time, but we do a fixed number of high-quality starts
    # and use the remaining time for the best one's local search?
    # Or just run GRASP for the entire time.
    # Let's run GRASP iterations. Each iteration: Construct + Local Search.
    # If Local Search is expensive, we might only do a few iterations.
    # Given n is likely small (< 100), Local Search is fast.
    
    while time.time() < deadline:
        # 1. Constructive Phase: Randomized Greedy (RCL)
        # Sort by total processing time
        totals = processing_times.sum(axis=1)
        # Create a candidate list of top k% jobs
        k = max(1, int(n * 0.1)) # Top 10%
        
        # We will build the sequence step by step.
        # At each step, choose from the top k jobs based on a greedy criterion (makespan increase)
        # But calculating makespan for every candidate at every step is O(n^3 * m).
        # For small n, this is fine.
        
        remaining_jobs = list(range(n))
        current_perm = []
        
        # To speed up, we can use a simpler randomized construction:
        # Pick a random job from the top k% of the remaining jobs sorted by total processing time.
        # This is a standard GRASP construction heuristic.
        
        while remaining_jobs:
            # Sort remaining jobs by total processing time
            remaining_sorted = sorted(remaining_jobs, key=lambda j: -totals[j])
            # Pick a random job from the top k
            k_actual = min(k, len(remaining_sorted))
            chosen = random.choice(remaining_sorted[:k_actual])
            current_perm.append(chosen)
            remaining_jobs.remove(chosen)
            
        current_perm = np.array(current_perm)
        current_mk = _makespan(current_perm, processing_times)
        
        # 2. Local Search Phase: VND
        refined_perm, refined_mk = _local_search_vnd(current_perm, processing_times, deadline)
        
        if refined_mk < best_mk_overall:
            best_mk_overall = refined_mk
            best_perm_overall = refined_perm.copy()
            
        # If we've found a very good solution, we might stop early?
        # No, let's keep trying to find a better start.
        
        # Optimization: If the local search didn't improve much, we might try a different k?
        # For simplicity, we keep k fixed.
        
        # To avoid infinite loops if local search is very fast, we add a small sleep or check time.
        # The while loop condition handles the time limit.
        
    return list(best_perm_overall)
# EVOLVE-BLOCK-END
