"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    # We only need the last row.
    # C_prev[k] is the completion time of the previous job on machine k.
    # C_curr[k] is the completion time of the current job on machine k.
    
    C_prev = np.zeros(m)
    for j in perm:
        C_curr = np.zeros(m)
        for k in range(m):
            if k == 0:
                C_curr[0] = C_prev[0] + pt[j, 0]
            else:
                C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        C_prev = C_curr
        
    return C_prev[-1]

def _neh_heuristic(pt):
    """
    NEH heuristic for flow shop.
    Returns a permutation list.
    """
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
        
    # 1. Calculate total processing time for each job
    totals = pt.sum(axis=1)
    
    # 2. Sort jobs in non-increasing order of total processing time
    order = np.argsort(-totals)
    
    # 3. Start with the first job
    current_perm = [order[0]]
    
    # 4. Insert each subsequent job in the best position
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        
        # Try all possible insertion positions
        for pos in range(len(current_perm) + 1):
            # Create candidate permutation
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(new_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
                
        # Insert job at best position
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _local_search_insertion(perm, pt, deadline):
    """
    Local search using insertion moves (Or-opt style).
    Moves a job or a block of 1, 2, or 3 jobs to a different position.
    Returns the best permutation found.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = list(perm)
    current_mk = _makespan(current_perm, pt)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # Block sizes to consider
    block_sizes = [1, 2, 3]
    
    improved = True
    while improved:
        improved = False
        # Randomize block sizes for variety? Or fixed?
        # Let's iterate through all possible moves for each block size.
        # To avoid getting stuck in local minima, we can use a first-improvement or best-improvement strategy.
        # Best improvement is safer but slower. First improvement is faster.
        # Given the 120s limit, let's use a mix or first-improvement with a limit on restarts.
        
        # Let's do a full scan for best improvement in this iteration.
        best_move_found = False
        best_new_mk = current_mk
        best_new_perm = None
        
        for b_size in block_sizes:
            if b_size > n:
                continue
            for start in range(n - b_size + 1):
                # Extract block
                block = current_perm[start:start+b_size]
                # Remove block
                remaining = current_perm[:start] + current_perm[start+b_size:]
                
                # Try inserting block at every position in remaining
                for pos in range(len(remaining) + 1):
                    # Skip if it's the original position (or effectively same)
                    # Original position in remaining is start (if start < original start) or start (if start >= original start)
                    # Actually, if we remove at 'start', the position 'start' in the new list is the same slot.
                    # Inserting at 'start' puts it back.
                    if pos == start:
                        continue
                    if pos == start + b_size:
                        # This is also effectively the same relative order if we consider the gap?
                        # No, if we remove 2 items at index 2, and insert at index 2, it's the same.
                        # If we insert at index 2+b_size, it's also the same?
                        # Let's just compute makespan to be safe, or skip if pos is start or start+b_size?
                        # Actually, inserting at 'start' in the reduced list puts it back exactly.
                        # Inserting at 'start+b_size' puts it back exactly if we consider the list indices.
                        # Let's just compute.
                        pass
                    
                    new_perm = remaining[:pos] + block + remaining[pos:]
                    mk = _makespan(new_perm, pt)
                    
                    if mk < best_new_mk:
                        best_new_mk = mk
                        best_new_perm = new_perm
                        best_move_found = True
                        
        if best_move_found and best_new_perm is not None:
            current_perm = best_new_perm
            current_mk = best_new_mk
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm[:]
            improved = True
            
            # Check time
            if time.time() > deadline:
                break
        else:
            # No improvement found in this pass
            # To escape local optima, we could do a random perturbation and restart search?
            # Or just break.
            # Let's try a random perturbation if we have time.
            if time.time() < deadline - 1.0:
                # Perturb: swap two random jobs
                i, j = random.sample(range(n), 2)
                perturbed = current_perm[:]
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                p_mk = _makespan(perturbed, pt)
                
                # Accept perturbation if it's not too bad? Or just use it as new start?
                # Let's just restart the local search from the perturbed state if it's close.
                # Or simply break to save time for other strategies.
                # For now, let's break to avoid infinite loops in local optima.
                break
            else:
                break
                
    return best_perm

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    # Safety margin: 120s limit. Let's use 110s to be safe.
    deadline = start_time + 110.0
    
    # 1. Initial Solution via NEH
    current_perm = _neh_heuristic(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Local Search with Insertion Moves
    # We can run multiple iterations of local search with random restarts if time permits.
    # But a single deep local search is usually good.
    # Let's run local search.
    result = _local_search_insertion(current_perm, processing_times, deadline)
    if result:
        mk = _makespan(result, processing_times)
        if mk < best_mk:
            best_mk = mk
            best_perm = result[:]
            
    # 3. If time remains, try Simulated Annealing or more restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try a few random restarts with NEH + Local Search
        # Or just SA on the best found so far.
        # Let's do SA on the current best.
        
        # SA Parameters
        T = 10.0
        # Estimate scale
        total_p = processing_times.sum()
        if total_p > 0:
            T = total_p * 0.05
        cooling = 0.99
        min_T = 1e-6
        
        current_sa_perm = np.array(best_perm)
        current_sa_mk = best_mk
        
        while time.time() < deadline - 1.0:
            if T < min_T:
                break
                
            # Move: Swap two random positions
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
                
            candidate = current_sa_perm.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            cand_mk = _makespan(candidate, processing_times)
            delta = cand_mk - current_sa_mk
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_sa_perm = candidate
                current_sa_mk = cand_mk
                
                if current_sa_mk < best_mk:
                    best_mk = current_sa_mk
                    best_perm = current_sa_perm.copy()
                    
            T *= cooling
            
    return list(best_perm)
# EVOLVE-BLOCK-END
