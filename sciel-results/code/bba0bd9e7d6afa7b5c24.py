"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    """
    if not order:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
        
    curr_completion = [0.0] * n_machines
    
    for j in order:
        curr_completion[0] += processing_times[j, 0]
        for m in range(1, n_machines):
            if curr_completion[m-1] > curr_completion[m]:
                curr_completion[m] = curr_completion[m-1] + processing_times[j, m]
            else:
                curr_completion[m] = curr_completion[m] + processing_times[j, m]
                
    return curr_completion[-1]


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    job_sums = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _local_search_insertion(order, processing_times, deadline):
    """
    Perform insertion-based local search.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            job = current_order[i]
            current_order.pop(i)
            
            best_pos = i
            best_ms = float('inf')
            
            for pos in range(n):
                current_order.insert(pos, job)
                ms = _makespan(current_order, processing_times)
                
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
                
                current_order.pop(pos)
            
            current_order.insert(best_pos, job)
            
            if best_ms < current_ms:
                current_ms = best_ms
                improved = True
                
            if time.time() > deadline:
                break
                
    return current_order, current_ms


def _adjacent_swap_ls(order, processing_times, deadline):
    """
    Perform local search using only adjacent swaps.
    This is faster per iteration than insertion, allowing more iterations.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    if n == 1:
        return order, _makespan(order, processing_times)
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through all adjacent pairs
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            # Try swapping i and i+1
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            new_ms = _makespan(current_order, processing_times)
            
            if new_ms < current_ms:
                current_ms = new_ms
                improved = True
            else:
                # Swap back
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                
    return current_order, current_ms


def _simulated_annealing(order, processing_times, deadline, initial_temp=1000.0, cooling=0.999):
    """
    Perform Simulated Annealing to escape local minima.
    Uses adjacent swaps primarily.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_ms = current_ms
    
    temp = initial_temp
    
    while time.time() < deadline and temp > 1e-6:
        if random.random() < 0.8:
            i = random.randint(0, n - 2)
            j = i + 1
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_ms = _makespan(current_order, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0:
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        else:
            if random.random() < np.exp(-delta / temp):
                current_ms = new_ms
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        temp *= cooling
        
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local Search
    # Use insertion local search first to get into a good basin
    ls_deadline = start_time + (deadline - start_time) * 0.5
    
    current_order, current_makespan = _local_search_insertion(current_order, processing_times, ls_deadline)
    
    if current_makespan < best_makespan:
        best_makespan = current_makespan
        best_order = current_order[:]
        
    # 3. Adjacent Swap Local Search (faster, more iterations)
    if time.time() < deadline:
        swap_deadline = start_time + (deadline - start_time) * 0.8
        swap_order, swap_ms = _adjacent_swap_ls(current_order, processing_times, swap_deadline)
        
        if swap_ms < best_makespan:
            best_makespan = swap_ms
            best_order = swap_order[:]
            
    # 4. Simulated Annealing
    if time.time() < deadline:
        sa_order, sa_ms = _simulated_annealing(current_order, processing_times, deadline, initial_temp=500.0, cooling=0.9995)
        
        if sa_ms < best_makespan:
            best_makespan = sa_ms
            best_order = sa_order[:]
            
    return best_order

# EVOLVE-BLOCK-END
