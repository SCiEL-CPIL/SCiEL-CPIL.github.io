"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """
    Compute the makespan for a given permutation of job indices.
    Optimized for speed.
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


def _neh_initialization(processing_times):
    """
    NEH heuristic for initial permutation.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the position that minimizes the incremental makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Step 1: Sort by total processing time
    total_processing = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_processing)
    
    # Step 2: Insert one by one
    current_perm = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm.copy()
            new_perm.insert(pos, job)
            ms = _makespan(new_perm, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm.insert(best_pos, job)
        
    return current_perm


def _composite_key_initialization(processing_times):
    """
    Initialize permutation using composite key derived from longest path in job-machine dependency graph.
    This models the flow-shop as a single-machine scheduling problem with time-lags.
    The composite key for each job is computed as the sum of processing times weighted by machine index,
    which approximates the longest path through the dependency graph.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Compute composite key: weighted sum of processing times
    # Higher weights for later machines (they have more downstream dependencies)
    weights = np.arange(1, n_machines + 1, dtype=float)
    composite_keys = processing_times @ weights
    
    # Sort by composite key in descending order
    sorted_jobs = np.argsort(-composite_keys)
    
    # Refine with NEH insertion
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
    Tabu search with swaps and insertions.
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
        if n_jobs <= 30:
            moves_to_try = []
            for i in range(n_jobs - 1):
                moves_to_try.append(('swap', i, i + 1))
            for _ in range(80):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    moves_to_try.append(('move', i, j))
        else:
            moves_to_try = []
            for _ in range(300):
                if np.random.rand() < 0.5:
                    i = np.random.randint(0, n_jobs - 1)
                    moves_to_try.append(('swap', i, i + 1))
                else:
                    i = np.random.randint(0, n_jobs)
                    j = np.random.randint(0, n_jobs)
                    if i != j:
                        moves_to_try.append(('move', i, j))
        
        for move in moves_to_try:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                move_key = ('swap', min(i, j), max(i, j))
            else:
                i, j = move[1], move[2]
                if i == j:
                    continue
                move_key = ('move', i, j)
            
            # Check tabu
            if move_key in tabu_list and tabu_list[move_key] > iteration:
                continue
            
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
            
            # Aspiration criterion
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
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
            # Perturb with a random move
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                new_perm = current_perm.copy()
                job_to_move = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
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
    Uses composite key initialization and tabu search with aspiration.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: composite key based
    initial_perm = _composite_key_initialization(processing_times)
    
    # Refine with tabu search
    n_jobs = len(processing_times)
    if n_jobs <= 20:
        time_limit = 15.0
    elif n_jobs <= 50:
        time_limit = 10.0
    else:
        time_limit = 5.0
    
    final_perm = _tabu_search(initial_perm, processing_times, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
