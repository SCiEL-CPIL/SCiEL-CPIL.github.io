"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute the makespan for a given permutation of jobs."""
    n_machines = processing_times.shape[1]
    prev_completion = np.zeros(n_machines)
    for j in order:
        current_completion = np.zeros(n_machines)
        current_completion[0] = prev_completion[0] + processing_times[j, 0]
        for m in range(1, n_machines):
            current_completion[m] = max(current_completion[m - 1], prev_completion[m]) + processing_times[j, m]
        prev_completion = current_completion
    return prev_completion[-1]


def _v_shape_initial(processing_times):
    """
    Generate an initial V-shaped permutation using a heuristic.
    Jobs with high total processing time go to the ends (front and back).
    """
    n_jobs = processing_times.shape[0]
    totals = processing_times.sum(axis=1)
    # Sort by total processing time descending
    sorted_jobs = np.argsort(-totals)
    
    order = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    for job in sorted_jobs:
        if left <= right:
            # Place in front
            order[left] = job
            left += 1
        if left <= right:
            # Place in back
            order[right] = job
            right -= 1
    
    return order


def _swap_makespan(order, i, j, processing_times):
    """
    Compute the new makespan after swapping positions i and j in order.
    This is a full recomputation for simplicity, but could be optimized.
    """
    new_order = order.copy()
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return _makespan(new_order, processing_times)


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses V-shaped heuristic for initialization and Tabu Search for improvement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial V-shaped solution
    best_order = _v_shape_initial(processing_times)
    best_makespan = _makespan(best_order, processing_times)
    
    # Tabu Search parameters
    tabu_length = max(10, n_jobs // 2)
    no_improve_count = 0
    max_no_improve = 500
    current_order = best_order.copy()
    current_makespan = best_makespan
    
    # Tabu list: store (i, j) swaps
    tabu_list = []
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate candidate moves
        best_move_makespan = float('inf')
        best_move_i = -1
        best_move_j = -1
        
        # Try all possible swaps (or a subset for large instances)
        if n_jobs <= 50:
            # Try all pairs
            for i in range(n_jobs):
                for j in range(i + 1, n_jobs):
                    # Check tabu
                    is_tabu = (i, j) in tabu_list or (j, i) in tabu_list
                    
                    new_makespan = _swap_makespan(current_order, i, j, processing_times)
                    
                    # Aspiration criterion: accept tabu move if it improves best
                    if not is_tabu or new_makespan < best_makespan:
                        if new_makespan < best_move_makespan:
                            best_move_makespan = new_makespan
                            best_move_i = i
                            best_move_j = j
        else:
            # For larger instances, try a random subset of moves
            num_candidates = min(100, n_jobs * (n_jobs - 1) // 2)
            tried = set()
            for _ in range(num_candidates):
                i = np.random.randint(n_jobs)
                j = np.random.randint(n_jobs)
                if i == j or (i, j) in tried or (j, i) in tried:
                    continue
                tried.add((i, j))
                
                is_tabu = (i, j) in tabu_list or (j, i) in tabu_list
                
                new_makespan = _swap_makespan(current_order, i, j, processing_times)
                
                if not is_tabu or new_makespan < best_makespan:
                    if new_makespan < best_move_makespan:
                        best_move_makespan = new_makespan
                        best_move_i = i
                        best_move_j = j
        
        # No valid move found
        if best_move_i == -1:
            # Try any move (relax tabu)
            i = np.random.randint(n_jobs)
            j = np.random.randint(n_jobs)
            while i == j:
                j = np.random.randint(n_jobs)
            best_move_makespan = _swap_makespan(current_order, i, j, processing_times)
            best_move_i = i
            best_move_j = j
        
        # Apply the best move
        current_order[best_move_i], current_order[best_move_j] = current_order[best_move_j], current_order[best_move_i]
        current_makespan = best_move_makespan
        
        # Update tabu list
        tabu_list.append((best_move_i, best_move_j))
        if len(tabu_list) > tabu_length:
            tabu_list.pop(0)
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = current_order.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    return best_order
# EVOLVE-BLOCK-END
