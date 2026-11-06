"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _v_shape_initial(processing_times):
    """
    Generate an initial V-shaped permutation.
    Sort jobs by total processing time descending.
    Place the largest jobs at the start and end, filling inwards.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs <= 2:
        return list(range(n_jobs))
    
    totals = processing_times.sum(axis=1)
    sorted_indices = np.argsort(-totals)
    
    perm = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    for i in range(n_jobs):
        if i % 2 == 0:
            perm[left] = sorted_indices[i]
            left += 1
        else:
            perm[right] = sorted_indices[i]
            right -= 1
            
    return perm

def _makespan(processing_times, perm):
    """Calculate the makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    prev_row = np.zeros(n_machines)
    curr_row = np.zeros(n_machines)
    
    makespan = 0.0
    for j in perm:
        p = processing_times[j]
        curr_row[0] = prev_row[0] + p[0]
        for m in range(1, n_machines):
            val1 = prev_row[m]
            val2 = curr_row[m-1]
            if val1 > val2:
                curr_row[m] = val1 + p[m]
            else:
                curr_row[m] = val2 + p[m]
        makespan = curr_row[-1]
        prev_row, curr_row = curr_row, prev_row
        
    return makespan

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: Initial V-shaped solution
    current_perm = _v_shape_initial(processing_times)
    current_makespan = _makespan(processing_times, current_perm)
    
    best_perm = current_perm.copy()
    best_makespan = current_makespan
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Tabu Search Parameters
    tabu_tenure = 15
    min_tenure = 5
    max_tenure = 50
    
    # Tabu list: dictionary mapping position i (swap i, i+1) to iteration when it expires
    tabu_list = {}
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Clean up tabu list periodically
        if iteration % 10 == 0:
            for key in list(tabu_list.keys()):
                if tabu_list[key] <= iteration:
                    del tabu_list[key]
            
        best_non_tabu_i = -1
        best_non_tabu_makespan = float('inf')
        best_tabu_i = -1
        best_tabu_makespan = float('inf')
        
        # Evaluate all adjacent swaps
        for i in range(n_jobs - 1):
            # Swap in place
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            new_makespan = _makespan(processing_times, current_perm)
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            
            is_tabu = i in tabu_list and tabu_list[i] > iteration
            
            if not is_tabu:
                if new_makespan < best_non_tabu_makespan:
                    best_non_tabu_makespan = new_makespan
                    best_non_tabu_i = i
            else:
                if new_makespan < best_tabu_makespan:
                    best_tabu_makespan = new_makespan
                    best_tabu_i = i
        
        # Select move
        chosen_i = -1
        chosen_makespan = current_makespan
        
        if best_non_tabu_i != -1:
            # Always accept the best non-tabu move, even if it worsens the solution
            # This allows escaping local optima
            chosen_i = best_non_tabu_i
            chosen_makespan = best_non_tabu_makespan
            
            # Dynamic Tenure: if improving, decrease tenure (intensify); if worsening, increase (diversify)
            if chosen_makespan < current_makespan:
                tabu_tenure = max(tabu_tenure - 1, min_tenure)
            else:
                tabu_tenure = min(tabu_tenure + 1, max_tenure)
                
        elif best_tabu_i != -1:
            # All moves are tabu. Use aspiration criterion.
            if best_tabu_makespan < best_makespan:
                chosen_i = best_tabu_i
                chosen_makespan = best_tabu_makespan
            else:
                # If no aspiration, pick the best tabu move anyway to break symmetry
                chosen_i = best_tabu_i
                chosen_makespan = best_tabu_makespan
                # Increase tenure to allow more exploration?
                tabu_tenure = min(tabu_tenure + 2, max_tenure)
        else:
            # No moves? Should not happen if n_jobs > 1
            break
        
        # Apply the chosen move
        current_perm[chosen_i], current_perm[chosen_i+1] = current_perm[chosen_i+1], current_perm[chosen_i]
        current_makespan = chosen_makespan
        
        # Update Tabu List
        tabu_list[chosen_i] = iteration + tabu_tenure
        
        # Update Best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_perm = current_perm.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
