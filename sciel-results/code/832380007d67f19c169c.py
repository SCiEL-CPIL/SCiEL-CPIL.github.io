"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    Optimized for small N (jobs) and M (machines).
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    
    if n_machines == 1:
        return np.sum(processing_times[perm, 0])
        
    # Standard loop
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = processing_times[job_idx]
        prev_c = 0.0
        for m in range(n_machines):
            # c_prev = completion[m] (completion of previous job on this machine)
            # c_left = prev_c (completion of current job on previous machine)
            # new_c = max(c_prev, c_left) + row[m]
            if completion[m] > prev_c:
                new_c = completion[m] + row[m]
            else:
                new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
            
    return completion[-1]

def _v_shape_heuristic(processing_times):
    """
    Construct an initial solution using the V-shaped heuristic.
    Jobs with shorter total processing times are placed in the middle,
    and longer ones at the ends. This is known to perform well for flow shops.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    # Sort by total processing time descending
    sorted_jobs = np.argsort(-totals).tolist()
    
    perm = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    
    # We alternate placing the longest remaining job on the left or right
    # to create a V-shape (long at ends, short in middle)
    # Actually, standard V-shape often places shortest in the middle.
    # Let's try: Place jobs in order of increasing total processing time into the middle?
    # Or decreasing?
    # Literature suggests: Sort by total processing time. 
    # A common V-shaped construction:
    # 1. Sort jobs by total processing time ascending.
    # 2. Place the job with the smallest total in the middle.
    # 3. Place the next smallest to the left or right (whichever is shorter/better).
    # 4. Repeat.
    
    # Let's implement a simple version:
    # Sort by total processing time ascending.
    sorted_asc = np.argsort(totals).tolist()
    
    perm = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    
    for job in sorted_asc:
        if left > right:
            break
        # Decide where to put the next shortest job
        # Usually we fill from the outside in? No, V-shape means short in middle.
        # So we should place the shortest jobs in the middle positions.
        
        # Let's try a different approach:
        # Sort by total processing time DESCENDING.
        # Place longest at start, next longest at end, next at start+1, etc.
        
        pass

    # Let's use a simpler heuristic that is often competitive: NEH is good, but let's try a "V" based on totals.
    # Actually, the prompt asks for V-shaped.
    # Algorithm:
    # 1. Calculate total processing time for each job.
    # 2. Sort jobs by total processing time in descending order.
    # 3. Start with an empty sequence.
    # 4. For each job in sorted order:
    #    - If the sequence is empty, add it.
    #    - Else, add it to the left or right of the current sequence such that the makespan of the partial sequence is minimized.
    
    sorted_desc = np.argsort(-totals).tolist()
    current_perm = [sorted_desc[0]]
    
    for i in range(1, n_jobs):
        job = sorted_desc[i]
        
        # Try adding to left
        perm_left = [job] + current_perm
        ms_left = _calc_makespan(processing_times, perm_left)
        
        # Try adding to right
        perm_right = current_perm + [job]
        ms_right = _calc_makespan(processing_times, perm_right)
        
        if ms_left < ms_right:
            current_perm = perm_left
        else:
            current_perm = perm_right
            
    return current_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: V-shaped Heuristic
    current_perm = _v_shape_heuristic(processing_times)
    best_perm = current_perm.copy()
    current_ms = _calc_makespan(processing_times, current_perm)
    best_ms = current_ms
    
    # 2. Tabu Search
    # Tabu list stores (i, j) pairs for swap moves to prevent cycling
    # Dynamic tabu tenure: start with a base tenure, adjust if stuck
    
    base_tenure = max(5, n_jobs // 2)
    current_tenure = base_tenure
    tabu_list = {} # key: (i, j) or frozenset({i,j}), value: iteration when it becomes non-tabu
    iteration = 0
    max_iterations = 100000 # Cap to avoid infinite loop if time check is slow, but time check is primary
    
    # Neighborhood: Swap two jobs at positions i and j
    # To speed up, we can restrict i and j to be close, or just random.
    # Random swap is O(1) to generate, O(N*M) to evaluate.
    
    no_improve_count = 0
    
    while (time.time() < deadline - 2.0) and iteration < max_iterations:
        
        # Generate a random swap
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        # Ensure i < j for consistent key
        if i > j:
            i, j = j, i
            
        key = (i, j)
        
        # Check if move is tabu
        is_tabu = key in tabu_list and tabu_list[key] > iteration
        
        # Calculate new makespan
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _calc_makespan(processing_times, new_perm)
        
        # Determine if move is acceptable
        # Accept if:
        # 1. It improves the current solution (new_ms < current_ms)
        # 2. It is not tabu
        # OR
        # 3. It improves the best solution found so far (Aspiration criterion)
        
        if new_ms < best_ms:
            # Aspiration: Always accept if it beats global best
            accept = True
        elif new_ms < current_ms and not is_tabu:
            # Improving and not tabu
            accept = True
        elif not is_tabu and random.random() < 0.1:
            # Allow some non-improving moves if not tabu to escape local minima
            # This adds a stochastic element similar to SA but constrained by tabu
            accept = True
        else:
            accept = False
            
        if accept:
            current_perm = new_perm
            current_ms = new_ms
            
            # Update tabu list
            # The move becomes tabu for the next 'current_tenure' iterations
            tabu_list[key] = iteration + current_tenure
            
            # Clean up old tabu entries to save memory (optional, dict is small)
            # if len(tabu_list) > 1000:
            #     for k in list(tabu_list.keys()):
            #         if tabu_list[k] <= iteration:
            #             del tabu_list[k]
            
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = current_perm.copy()
                no_improve_count = 0
                # If we found a new best, we might want to reset or adjust tenure?
                # Often, reducing tenure helps intensify around the new best.
                # current_tenure = max(5, current_tenure * 0.9)
            else:
                no_improve_count += 1
                
            # Dynamic tenure adjustment
            if no_improve_count > 50:
                # Increase tenure to diversify (explore more distant areas)
                current_tenure = min(n_jobs, current_tenure * 1.2)
                no_improve_count = 0
            elif no_improve_count > 20:
                # Slightly decrease to intensify? Or just keep.
                pass
                
        iteration += 1
        
        # Periodic cleanup of tabu list
        if iteration % 1000 == 0:
            for k in list(tabu_list.keys()):
                if tabu_list[k] <= iteration:
                    del tabu_list[k]

    return best_perm
# EVOLVE-BLOCK-END
