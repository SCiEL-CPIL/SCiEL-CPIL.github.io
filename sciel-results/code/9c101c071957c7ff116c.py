"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _initial_completion(perm, pt):
    """
    Compute the full completion time matrix C[j, m] for a permutation.
    C[j, m] is the completion time of job perm[j] on machine m.
    Returns a 2D numpy array of shape (n, m).
    """
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return np.zeros((n, m))
    
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[perm[j], m]
    # We can compute this row by row.
    # To make it fast, we pre-extract the processing times for the permutation.
    # pt_perm has shape (n, m) where pt_perm[j, m] = pt[perm[j], m]
    pt_perm = pt[perm]
    
    C = np.zeros((n, m))
    # First job
    C[0, 0] = pt_perm[0, 0]
    for m_idx in range(1, m):
        C[0, m_idx] = C[0, m_idx-1] + pt_perm[0, m_idx]
        
    for j_idx in range(1, n):
        C[j_idx, 0] = C[j_idx-1, 0] + pt_perm[j_idx, 0]
        for m_idx in range(1, m):
            # C[j_idx, m_idx] = max(C[j_idx-1, m_idx], C[j_idx, m_idx-1]) + pt_perm[j_idx, m_idx]
            # Using max is fine for small m, but we can vectorize the inner loop if we were doing rows.
            # Since we are filling row by row, we just do the loop.
            C[j_idx, m_idx] = max(C[j_idx-1, m_idx], C[j_idx, m_idx-1]) + pt_perm[j_idx, m_idx]
            
    return C

def _delta_swap(perm, C, pt, i, j):
    """
    Calculate the change in makespan for swapping jobs at positions i and j (i < j).
    Uses incremental update on the completion matrix.
    Returns (delta, new_C) where new_C is the updated completion matrix.
    """
    n = len(perm)
    m = pt.shape[1]
    
    if i == j:
        return 0.0, C
    
    # We need to update C for rows i through j.
    # Rows < i are unchanged.
    # Rows > j are affected because their predecessors changed.
    
    # To avoid copying the whole matrix every time (which is O(n*m)), 
    # we can copy it, but n is likely small enough (< 100) that O(n*m) copy is fast.
    # However, we can be smarter. Let's just copy for simplicity and correctness first, 
    # then optimize if needed. Given 120s limit and n~50-100, O(n*m) is ~1000 ops.
    # Tabu search might do 100k+ moves. 100k * 1000 = 100M ops. Python loop is ~100M ops/s? 
    # No, Python loops are ~1M-10M ops/s. 100M ops would take 10-100s. Too slow.
    
    # We need a faster delta calculation or a faster update.
    # Actually, the standard trick is that only the segment [i, j] changes.
    # But rows > j also change because C[j, m] changes.
    # Wait, C[j+1, m] depends on C[j, m]. So yes, all rows > j change.
    # This makes O(n) delta hard for flow shop without more complex math.
    
    # Alternative: Just recalculate makespan from scratch? O(n*m).
    # If we do 10000 moves, and n*m=1000, that's 10M ops. That's fine.
    # The bottleneck is the Python loop in _makespan.
    
    # Let's try to make _makespan as fast as possible using NumPy vectorization where possible.
    # The dependency C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m] is sequential in j.
    # But for a fixed j, we can compute the row if we have the previous row.
    
    # Let's stick to a very tight loop for makespan calculation.
    pass

def _fast_makespan(perm, pt):
    """
    Optimized makespan calculation.
    """
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # Extract processing times for the permutation
    # pt[perm] creates a new array. This is O(n*m) memory copy.
    # We can avoid this by indexing pt directly in the loop, which is slower per access.
    # Let's benchmark mentally: 
    # Option A: pt_p = pt[perm]; then loop over pt_p.
    # Option B: Loop over perm, access pt[perm[j]].
    # Option A is generally faster in NumPy due to contiguous memory and vectorized indexing overhead being paid once.
    
    pt_p = pt[perm]
    
    # We only need the last column of C to get makespan? 
    # No, C[j, m] depends on C[j, m-1].
    # We can maintain a 1D array `prev_row` representing C[j-1, :].
    # And `curr_row` for C[j, :].
    
    prev_row = np.zeros(m)
    
    for j in range(n):
        p_j = pt_p[j]
        # curr_row[0] = prev_row[0] + p_j[0]
        # curr_row[k] = max(prev_row[k], curr_row[k-1]) + p_j[k]
        
        # Vectorization attempt:
        # curr_row[0] is easy.
        # curr_row[1:] = max(prev_row[1:], cumsum(p_j[1:]) + curr_row[0]... ) No, that's not right.
        # The recurrence is:
        # C_j_m = max(C_{j-1, m}, C_{j, m-1}) + p_{j, m}
        # This is a classic DP. Hard to fully vectorize across m for a single j without iterating m.
        # But m is usually small (5-10). n is larger (20-100).
        
        # Let's just loop m.
        curr_row = np.empty(m)
        curr_row[0] = prev_row[0] + p_j[0]
        for k in range(1, m):
            # max(prev_row[k], curr_row[k-1])
            if prev_row[k] > curr_row[k-1]:
                curr_row[k] = prev_row[k] + p_j[k]
            else:
                curr_row[k] = curr_row[k-1] + p_j[k]
        
        prev_row = curr_row
        
    return prev_row[-1]

def _neh_heuristic(pt):
    """
    NEH heuristic for initial solution.
    """
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
        
    # Sort by total processing time descending
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    current_perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        
        # Try all positions
        # Optimization: We can break early if we find a position that is clearly bad?
        # No, just do all. n is small.
        for pos in range(len(current_perm) + 1):
            # Construct new perm
            # To avoid list slicing overhead in tight loop, we can just calculate makespan
            # on the fly? No, we need the perm.
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _fast_makespan(new_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Generate initial solution using NEH
    best_perm = _neh_heuristic(processing_times)
    best_mk = _fast_makespan(best_perm, processing_times)
    
    # 2. Tabu Search
    # Parameters
    tabu_tenure = 10  # Length of tabu list
    tabu_list = []    # List of (i, j) pairs, or just positions? 
                      # For swap moves, we can store (min(i,j), max(i,j))
    
    # We will use a simple tabu list of moves (i, j) with i < j.
    # To check if a move is tabu, we check if (i, j) is in the list.
    # For efficiency, we can use a set or a list. List is fine for small tenure.
    
    current_perm = best_perm[:]
    current_mk = best_mk
    
    # Diversification: If no improvement for X moves, restart from best or random?
    # Let's stick to simple TS first.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Generate candidate moves: swap i and j
        # To ensure we explore well, we can pick a random i and j.
        # Or pick a random i and a random j within a window?
        # Global swaps are good for diversification.
        
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
            
        move = (i, j)
        
        # Check tabu
        # If the move is in the tabu list, it is forbidden UNLESS it leads to a solution better than best_mk (aspiration criterion)
        is_tabu = False
        if len(tabu_list) > 0:
            # Simple check. For larger tenure, use a set.
            # Since tenure is small, list is OK.
            if move in tabu_list:
                is_tabu = True
        
        # Evaluate candidate
        candidate = current_perm[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_mk = _fast_makespan(candidate, processing_times)
        delta = cand_mk - current_mk
        
        # Acceptance criterion
        accept = False
        if delta < 0:
            # Improvement
            accept = True
            # Even if tabu, we accept if it's better than global best? 
            # Standard TS: Tabu is overridden by aspiration (cand_mk < best_mk).
            if is_tabu and cand_mk >= best_mk:
                accept = False
        else:
            # Non-improvement. Only accept if NOT tabu.
            if not is_tabu:
                # In standard TS, we might accept non-improving moves to escape local minima.
                # However, for flow shop, hill-climbing with tabu (only accepting improvements) 
                # often gets stuck. 
                # Let's try accepting non-improving moves if they are not tabu? 
                # That turns it into a random walk with memory. 
                # Better: Accept non-improving moves with some probability? 
                # Or just strict hill climbing with tabu? 
                # Let's try strict hill climbing first, but if stuck, we need diversification.
                # For now, let's accept non-improving moves if not tabu to allow exploration.
                accept = True
                # Actually, pure hill climbing with tabu is "Tabu Hill Climbing".
                # Let's try that. If we accept non-improving, it's "Tabu Search" in the broader sense.
                # Let's stick to: Accept if (delta < 0) OR (delta >= 0 AND NOT is_tabu).
                # This allows wandering.
        
        if accept:
            # Update current
            current_perm = candidate
            current_mk = cand_mk
            
            # Update tabu list
            tabu_list.append(move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
                
            # Update best
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Diversification strategy
            if no_improve_count > max_no_improve:
                # Restart from best solution with some perturbation
                current_perm = best_perm[:]
                current_mk = best_mk
                # Perturb: swap 3 random pairs
                for _ in range(3):
                    k, l = random.sample(range(n), 2)
                    current_perm[k], current_perm[l] = current_perm[l], current_perm[k]
                current_mk = _fast_makespan(current_perm, processing_times)
                tabu_list = []
                no_improve_count = 0
                
    return best_perm

# EVOLVE-BLOCK-END
