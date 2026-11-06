"""
Strategy: Best-Insertion Construction + Tabu Search with Block-Insertion Neighborhood.
This differs from the parent (NEH + random perturbation + adjacent swap) by using a 
deterministic, greedy construction and a memory-based metaheuristic (Tabu Search) 
with a much larger neighborhood (block moves) to escape local minima that adjacent 
swaps cannot.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    completion = np.zeros(n_machines)
    for job_idx in perm:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job_idx, m]
            else:
                completion[m] = max(completion[m], completion[m - 1]) + processing_times[job_idx, m]
    return completion[-1]

def _best_insertion_construction(processing_times):
    """
    Construct an initial permutation using Best-Insertion heuristic.
    Jobs are sorted by total processing time (descending).
    Each job is inserted at the position that minimizes the makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Sort jobs by total processing time (descending)
    totals = processing_times.sum(axis=1)
    job_order = np.argsort(-totals)
    
    perm = []
    for job in job_order:
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at each position
        for pos in range(len(perm) + 1):
            # Create new permutation
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert job at best position
        perm = perm[:best_pos] + [job] + perm[best_pos:]
        
    return perm

def _tabu_search(perm, processing_times, max_iterations=5000, tabu_tenure=7, deadline=None):
    """
    Tabu Search with Block-Insertion neighborhood.
    Moves: Select a block of length L (1, 2, or 3) and move it to a different position.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Tabu list: stores (start, end, new_pos) -> iteration when it becomes non-tabu
    # We'll use a dictionary for O(1) lookup
    tabu_dict = {}
    iteration = 0
    
    # Block lengths to consider
    block_lengths = [1, 2, 3]
    
    while iteration < max_iterations:
        if deadline is not None and time.time() > deadline:
            break
            
        best_move = None
        best_move_ms = current_ms
        best_move_key = None
        aspiration_met = False
        
        # Generate neighbors
        for block_len in block_lengths:
            if block_len >= n_jobs:
                continue
                
            # Start positions for the block
            for start in range(n_jobs - block_len + 1):
                end = start + block_len - 1
                
                # Skip moving to the same position
                # Destination positions: 0 to n_jobs - block_len
                # But not the original position 'start'
                for dest in range(n_jobs - block_len + 1):
                    if dest == start:
                        continue
                    
                    # Define the move: remove block [start:end+1] and insert at dest
                    # If dest < start, the block moves left. If dest > end, the block moves right.
                    
                    # Create new permutation
                    block = current_perm[start:end+1]
                    remaining = current_perm[:start] + current_perm[end+1:]
                    
                    # Adjust dest if we removed elements before it
                    if dest > start:
                        new_dest = dest - block_len
                    else:
                        new_dest = dest
                    
                    new_perm = remaining[:new_dest] + block + remaining[new_dest:]
                    
                    ms = _makespan(new_perm, processing_times)
                    
                    # Check tabu status
                    # Key: (start, end, new_dest)
                    move_key = (start, end, new_dest)
                    is_tabu = move_key in tabu_dict and tabu_dict[move_key] > iteration
                    
                    # Aspiration criterion: accept if it improves the best solution found so far
                    if ms < best_ms:
                        aspiration_met = True
                    else:
                        aspiration_met = False
                    
                    # Accept if not tabu, or if aspiration is met
                    if (not is_tabu) or aspiration_met:
                        if ms < best_move_ms:
                            best_move_ms = ms
                            best_move = new_perm
                            best_move_key = move_key
                            # Store original start/end/dest for the move
                            best_move_orig = (start, end, new_dest)
        
        # If no improving move found, pick the best non-tabu move (or best move if all tabu)
        # The loop above already finds the best non-tabu/aspiration move.
        # However, standard TS usually allows non-improving moves to escape local optima.
        # The logic above selects the best *feasible* (non-tabu or aspiration) move, 
        # which might be worse than current_ms. This is correct for TS.
        
        if best_move is None:
            # No valid moves? (Should not happen unless n_jobs is very small or tabu is too restrictive)
            break
            
        # Update tabu list
        # The move we just made: best_move_orig
        # It becomes non-tabu after tabu_tenure iterations
        tabu_dict[best_move_orig] = iteration + tabu_tenure
        
        # Clean up old tabu entries to save memory
        if len(tabu_dict) > 1000:
            keys_to_remove = [k for k, v in tabu_dict.items() if v <= iteration]
            for k in keys_to_remove:
                del tabu_dict[k]
        
        # Apply the move
        current_perm = best_move
        current_ms = best_move_ms
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
        
        iteration += 1
        
    return best_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Best-Insertion + Tabu Search.
    """
    start_time = time.time()
    # Safety margin: 145s out of 150s limit
    deadline = start_time + 145.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Best-Insertion Construction
    perm = _best_insertion_construction(processing_times)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    # Step 2: Tabu Search
    # Determine number of iterations based on problem size
    # For small n, we can do more. For large n, fewer but each iteration is more expensive.
    # Let's aim for a fixed number of iterations that fits in time, or dynamic.
    # _makespan is O(n*m). Block insertion neighborhood is O(n^2 * L).
    # Total per iteration: O(n^3 * L * m).
    # If n=20, m=10, L=3: 20^3 * 3 * 10 = 240,000 ops. 5000 iters = 1.2e9 ops. Might be too slow in Python.
    # Let's optimize the inner loop or reduce iterations.
    
    # Actually, _makespan in Python is slow.
    # Let's check time for one iteration.
    # If n=10, m=10. Neighborhood size ~ 10*10*3 = 300.
    # 300 * 10*10 = 30,000 ops.
    # 5000 iters = 150,000,000 ops. Should be fine.
    
    # If n=50, m=10. Neighborhood size ~ 50*50*3 = 7500.
    # 7500 * 50*10 = 3,750,000 ops.
    # 5000 iters = 18,750,000,000 ops. Too slow.
    
    # So we need to adapt iterations to n.
    # Let's estimate ops per iteration ~ n^3 * m.
    # Target total ops ~ 1e9 (approx 1-2 seconds in Python? No, Python is slower. ~1e8-1e9 simple ops/s).
    # Let's say we want to use 140 seconds.
    # We can do a few iterations to estimate speed, then set max_iterations.
    
    # Simple approach:
    # If n < 15: max_iters = 5000
    # If n < 30: max_iters = 1000
    # If n < 50: max_iters = 200
    # Else: max_iters = 50
    
    if n_jobs < 15:
        max_iters = 5000
    elif n_jobs < 30:
        max_iters = 1000
    elif n_jobs < 50:
        max_iters = 200
    else:
        max_iters = 50
        
    perm_ts = _tabu_search(perm, processing_times, max_iterations=max_iters, tabu_tenure=7, deadline=deadline)
    ms_ts = _makespan(perm_ts, processing_times)
    
    if ms_ts < best_ms:
        best_perm = list(perm_ts)
        best_ms = ms_ts
        
    return best_perm
# EVOLVE-BLOCK-END
