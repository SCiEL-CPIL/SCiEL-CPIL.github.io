"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _v_shape_initial(processing_times):
    """
    Generate an initial V-shaped permutation.
    
    Heuristic: Sort jobs by total processing time.
    The job with the largest total time goes to position 0.
    The next largest goes to the last position.
    The next goes to position 1.
    The next goes to second to last.
    And so on. This creates a V-shape (or U-shape) where high-total jobs are at the ends.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    totals = processing_times.sum(axis=1)
    sorted_indices = np.argsort(totals)
    
    # We want largest at ends.
    # Let's pick from the end of the sorted list (largest) for the "ends" of the permutation.
    # We fill the permutation from both ends towards the center.
    
    perm = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    # sorted_indices is ascending. We traverse from largest to smallest.
    # We alternate placing the current largest at left, then right, etc.
    # Actually, a common V-shape puts the biggest at the start, then fills the rest in a way that creates the V.
    # Let's try: 
    # 1. Find index of max total. Place at 0.
    # 2. From remaining, find max total. Place at end.
    # 3. From remaining, find max total. Place at 1.
    # 4. From remaining, find max total. Place at end-1.
    
    remaining = list(sorted_indices[::-1]) # Descending order of totals
    
    # We will fill positions 0, n-1, 1, n-2, ...
    # If we have 5 jobs: 0, 4, 1, 3, 2
    # Job at 0: Max
    # Job at 4: Next Max
    # Job at 1: Next Max
    # Job at 3: Next Max
    # Job at 2: Min
    
    pos = 0
    while remaining:
        job = remaining.pop(0) # Largest remaining
        if pos % 2 == 0:
            perm[left] = job
            left += 1
        else:
            perm[right] = job
            right -= 1
        pos += 1
        
    return perm

def _compute_makespan(processing_times, perm):
    """
    Compute the makespan for a given permutation.
    Returns the makespan and the completion time matrix (for potential reuse, though we will use incremental updates in TS).
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0 or n_machines == 0:
        return 0.0, np.zeros((0, 0))
    
    # C[j][m] is completion time of job j on machine m
    # We compute row by row.
    # C[j][0] = C[j-1][0] + p[j][0]
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    
    # We only need the final row for makespan, but for incremental updates, 
    # we need the "state" at each step.
    # For the initial calculation, just return makespan.
    
    prev_row = np.zeros(n_machines)
    curr_row = np.zeros(n_machines)
    
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                curr_row[0] = prev_row[0] + processing_times[j, 0]
            else:
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + processing_times[j, m]
        prev_row = curr_row.copy()
        
    makespan = prev_row[-1]
    return makespan

def _eval_neighbors_incremental(processing_times, perm, n_jobs, n_machines):
    """
    Efficiently evaluate the makespan change for all adjacent swaps.
    
    Returns a list of (new_makespan, swap_index) for all i in 0..n_jobs-2.
    
    Method:
    1. Compute forward completion times F[j][m]: completion time of job at perm[j] on machine m.
    2. Compute backward completion times B[j][m]: "start time" or similar metric from the end.
       Specifically, B[j][m] is the time at which the job at perm[j] *could* start on machine m 
       if all subsequent jobs were scheduled optimally backwards? 
       
       Actually, a standard technique for adjacent swap evaluation in flow shop:
       Let C[j][m] be the completion time of job j on machine m.
       If we swap job at i and i+1, only the completion times for jobs i and i+1 and potentially subsequent jobs change?
       No, in flow shop, changing the order of job i and i+1 affects the start times of job i+1 and all subsequent jobs on all machines?
       Yes, it propagates. So we cannot just look at local changes unless we use a specific property.
       
       However, there is a known $O(N \cdot M)$ algorithm to evaluate all neighbors if we precompute:
       1. Forward matrix C_f[j][m]
       2. Backward matrix C_b[j][m] where C_b[j][m] represents the "earliest start time" of job j on machine m 
          assuming all jobs after j are fixed in their current order?
          
       Let's use the property:
       For a swap at i (swapping job A=perm[i] and B=perm[i+1]):
       The new makespan is determined by the critical path.
       
       Alternative: Since N is likely small (typically < 100 in these tasks), $O(N^2 \cdot M)$ per iteration might be too slow if we do many iterations.
       But $O(N \cdot M)$ per neighbor is $O(N^2 \cdot M)$ per iteration.
       If N=50, M=10, that's 2500 * 10 = 25,000 ops per iteration. 120s allows ~10^7 ops easily.
       So even full recomputation per neighbor ($O(N \cdot M)$) is $O(N^2 \cdot M)$ per iteration.
       Wait, full recomputation is $O(N \cdot M)$.
       If we do this for N neighbors, it is $O(N^2 \cdot M)$.
       For N=100, M=10, that is 10000 * 10 = 100,000 ops per iteration.
       120s / 100,000 = 1200 iterations.
       This is feasible.
       
       The current program's bottleneck might be Python overhead in the loop.
       Let's stick to full recomputation but vectorize or optimize the inner loop.
       Or better, implement the $O(N \cdot M)$ neighbor evaluation.
       
       The $O(N \cdot M)$ neighbor evaluation method:
       Let $C_f[i][m]$ be the completion time of the job at position $i$ on machine $m$ in the current sequence.
       Let $C_b[i][m]$ be the "backward completion time". 
       $C_b[i][m]$ is defined such that if we remove the job at $i$, the time available is related to $C_b$.
       
       Actually, let's use a simpler approach that is still faster than pure Python loops:
       Use NumPy to compute the makespan for the current permutation.
       For each neighbor, we can compute the makespan by modifying the sequence and recomputing.
       To speed this up, we can precompute the "prefix" and "suffix" states?
       
       Let's try the $O(N \cdot M)$ neighbor evaluation algorithm described in literature (e.g., by Taillard or others).
       
       1. Compute $C_f[j][m]$ for all $j, m$.
       2. Compute $C_b[j][m]$ backwards:
          $C_b[n-1][m] = 0$ ? No.
          $C_b[j][m] = \min(C_b[j+1][m+1], C_b[j+1][m]) - p_{j+1, m}$? No.
          
          Let's define $S_f[j][m]$ as start time of job $j$ on machine $m$.
          $S_f[0][0] = 0$.
          $S_f[j][0] = S_f[j-1][0] + p_{j-1, 0}$.
          $S_f[j][m] = \max(S_f[j-1][m] + p_{j-1, m}, S_f[j][m-1] + p_{j, m-1})$.
          $C_f[j][m] = S_f[j][m] + p_{j, m}$.
          
          For backward:
          $S_b[j][m]$ is the start time of job $j$ on machine $m$ if we schedule backwards from the end.
          $S_b[n-1][M-1] = 0$ (relative to makespan).
          $S_b[j][M-1] = S_b[j+1][M-1] + p_{j+1, M-1}$.
          $S_b[j][m] = \min(S_b[j+1][m] + p_{j+1, m}, S_b[j][m+1] + p_{j, m+1})$.
          
          If we swap job $i$ and $i+1$, the new makespan is:
          $\max( C_f[i-1][m] + \text{time to process } i, i+1 \text{ on } m, \dots )$
          
          This is complex. Given the time limit, I will implement a highly optimized full-recomputation using NumPy where possible, or just efficient Python loops.
          The previous program used Python loops for makespan. I will optimize `_makespan` using a 1D array trick which is already there.
          
          To improve speed, I will reduce the number of iterations by using a better initial solution and a more effective tabu search.
          I will also add a "restart" mechanism if stuck.
    """
    pass

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Initial V-Shaped Solution
    current_perm = _v_shape_initial(processing_times)
    current_makespan = _compute_makespan(processing_times, current_perm)
    
    best_perm = current_perm.copy()
    best_makespan = current_makespan
    
    # 2. Tabu Search
    start_time = time.time()
    time_limit = 115.0
    
    # Tabu list: dict of {swap_index: iteration_until_which_tabu}
    tabu_list = {}
    base_tabu_tenure = 10
    max_tabu_tenure = 30
    min_tabu_tenure = 5
    current_tabu_tenure = base_tabu_tenure
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Evaluate all neighbors
        best_move_i = -1
        best_move_makespan = current_makespan
        best_move_is_tabu = False
        
        # Track best tabu move for aspiration
        best_tabu_move_i = -1
        best_tabu_move_makespan = current_makespan
        
        for i in range(n_jobs - 1):
            # Create new perm
            new_perm = current_perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            
            new_makespan = _compute_makespan(processing_times, new_perm)
            
            is_tabu = (i in tabu_list) and (tabu_list[i] > iteration)
            
            if not is_tabu:
                if new_makespan < best_move_makespan:
                    best_move_makespan = new_makespan
                    best_move_i = i
                    best_move_is_tabu = False
            else:
                if new_makespan < best_tabu_move_makespan:
                    best_tabu_move_makespan = new_makespan
                    best_tabu_move_i = i
        
        # Select move
        if best_move_i != -1:
            chosen_i = best_move_i
            chosen_makespan = best_move_makespan
        elif best_tabu_move_i != -1:
            # Aspiration: accept if better than global best
            if best_tabu_move_makespan < best_makespan:
                chosen_i = best_tabu_move_i
                chosen_makespan = best_tabu_move_makespan
            else:
                # No good move, try a random move to diversify?
                # Or break.
                # Let's try a random non-tabu move if available, else break.
                # For now, break to avoid wasting time.
                break
        else:
            break
            
        # Apply move
        current_perm = current_perm.copy()
        current_perm[chosen_i], current_perm[chosen_i+1] = current_perm[chosen_i+1], current_perm[chosen_i]
        current_makespan = chosen_makespan
        
        # Update Tabu List
        tabu_list[chosen_i] = iteration + current_tabu_tenure
        
        # Update Best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_perm = current_perm.copy()
            no_improve_count = 0
            # Increase tenure slightly to stay in neighborhood
            current_tabu_tenure = min(current_tabu_tenure + 1, max_tabu_tenure)
        else:
            no_improve_count += 1
            # Decrease tenure to escape
            current_tabu_tenure = max(current_tabu_tenure - 1, min_tabu_tenure)
            
            # Diversification: if stuck for too long, restart with perturbation?
            # Simple perturbation: swap two random non-adjacent jobs
            if no_improve_count > 50:
                # Pick two random positions
                i = np.random.randint(0, n_jobs - 1)
                j = np.random.randint(0, n_jobs - 1)
                if i != j:
                    current_perm = current_perm.copy()
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    current_makespan = _compute_makespan(processing_times, current_perm)
                    no_improve_count = 0
                    current_tabu_tenure = base_tabu_tenure
                    # Clear tabu list to allow new directions
                    tabu_list.clear()
                    
        # Cleanup tabu list
        for key in list(tabu_list.keys()):
            if tabu_list[key] <= iteration:
                del tabu_list[key]
                
    return list(best_perm)
# EVOLVE-BLOCK-END
