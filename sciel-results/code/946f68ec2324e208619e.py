"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """
    Compute the makespan for a given permutation of job indices.
    Optimized for speed using Python lists and local variables.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    
    prev_job_completion = np.zeros(n_machines)
    
    for job in perm:
        current_job_completion_on_prev_machine = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                start = prev_job_completion[0]
                current_job_completion_on_prev_machine = start + p
            else:
                start = max(current_job_completion_on_prev_machine, prev_job_completion[m])
                current_job_completion_on_prev_machine = start + p
            prev_job_completion[m] = current_job_completion_on_prev_machine
    
    return prev_job_completion[-1]


def _composite_key_initialization(processing_times):
    """
    Initialize using composite key based on longest path in job-machine dependency graph.
    The composite key for each job is derived from:
    1. Total processing time
    2. Critical path length (longest path through the job in the dependency graph)
    3. Variance of processing times
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Calculate composite keys
    total_processing = processing_times.sum(axis=1)
    
    # For each job, compute a critical path-like measure
    # In a flow shop, the critical path for a job in isolation is just its total processing time
    # But we can consider the maximum processing time across machines as a bottleneck indicator
    max_processing = processing_times.max(axis=1)
    min_processing = processing_times.min(axis=1)
    
    # Composite key: weighted combination
    # Jobs with higher total processing should come first
    # Jobs with higher max processing (bottlenecks) should also be prioritized
    # Normalize each component
    if total_processing.max() > total_processing.min():
        norm_total = (total_processing - total_processing.min()) / (total_processing.max() - total_processing.min())
    else:
        norm_total = np.ones(n_jobs)
    
    if max_processing.max() > max_processing.min():
        norm_max = (max_processing - max_processing.min()) / (max_processing.max() - max_processing.min())
    else:
        norm_max = np.ones(n_jobs)
    
    # Variance as a measure of variability
    variance = processing_times.var(axis=1)
    if variance.max() > variance.min():
        norm_var = (variance - variance.min()) / (variance.max() - variance.min())
    else:
        norm_var = np.ones(n_jobs)
    
    # Composite key: prioritize high total, high max, low variance
    composite_key = 0.5 * norm_total + 0.3 * norm_max - 0.2 * norm_var
    
    # Sort by composite key in descending order
    sorted_jobs = np.argsort(-composite_key)
    
    # Now refine with insertion heuristic
    current_perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm.copy()
            new_perm.insert(pos, job)
            ms = _makespan(new_perm, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm.insert(best_pos, job)
    
    return current_perm


def _tabu_search(initial_perm, processing_times, time_limit=10.0):
    """
    Tabu search with swaps and insertions, with enhanced diversification.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Tabu list: dictionary mapping move tuple to expiration time
    tabu_list = {}
    tabu_tenure = 15
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 80
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        
        # Generate neighborhood
        moves_to_try = []
        
        # Adjacent swaps
        for i in range(n_jobs - 1):
            moves_to_try.append(('swap', i, i + 1))
        
        # Non-adjacent swaps and moves
        n_extra = min(100, n_jobs * 5)
        for _ in range(n_extra):
            if np.random.rand() < 0.4:
                # Swap
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    moves_to_try.append(('swap', i, j))
            else:
                # Move
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    moves_to_try.append(('move', i, j))
        
        for move in moves_to_try:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                if i == j:
                    continue
                move_key = ('swap', min(i, j), max(i, j))
            else:
                i, j = move[1], move[2]
                if i == j:
                    continue
                move_key = ('move', i, j)
            
            # Check tabu
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            
            # Generate new permutation
            new_perm = current_perm.copy()
            if move[0] == 'swap':
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                job_to_move = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
            
            new_makespan = _makespan(new_perm, processing_times)
            
            # Aspiration: accept if it improves best
            is_aspiration = new_makespan < best_makespan
            
            if is_tabu and not is_aspiration:
                continue
            
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_perm = new_perm
                best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            # Perturb: apply a random 2-opt style move
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                new_perm = current_perm.copy()
                if i > j:
                    i, j = j, i
                # Reverse segment
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                current_perm = new_perm
                current_makespan = _makespan(current_perm, processing_times)
        else:
            no_improve_count = 0
            tabu_list[best_neighbor_move] = iteration + tabu_tenure
            
            # Clean up tabu list
            for key in list(tabu_list.keys()):
                if tabu_list[key] <= iteration:
                    del tabu_list[key]
            
            current_perm = best_neighbor_perm
            current_makespan = best_neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses composite key initialization and tabu search with enhanced diversification.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: composite key based initialization
    initial_perm = _composite_key_initialization(processing_times)
    
    # Refine with tabu search
    n_jobs = len(processing_times)
    if n_jobs <= 20:
        time_limit = 18.0
    elif n_jobs <= 50:
        time_limit = 12.0
    else:
        time_limit = 8.0
    
    final_perm = _tabu_search(initial_perm, processing_times, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
