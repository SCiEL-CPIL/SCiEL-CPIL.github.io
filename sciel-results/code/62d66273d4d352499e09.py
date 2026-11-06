# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(order, processing_times):
    """
    Fast makespan calculation.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if len(order) == 0:
        return 0.0
    
    if n_machines == 1:
        return float(processing_times[order, 0].sum())
    
    prev = np.zeros(n_machines)
    pt = processing_times
    
    for j in order:
        row = pt[j]
        curr = np.empty(n_machines)
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            # Unroll max for speed
            if curr[m-1] > prev[m]:
                curr[m] = curr[m-1] + row[m]
            else:
                curr[m] = prev[m] + row[m]
        prev = curr
        
    return float(prev[-1])


def _makespan_incremental(order, processing_times, pos_remove, pos_insert):
    """
    Calculate makespan after removing job at pos_remove and inserting at pos_insert.
    Only recomputes from min(pos_remove, pos_insert) onwards.
    """
    n_machines = processing_times.shape[1]
    n = len(order)
    
    if n_machines == 0 or n == 0:
        return 0.0
    
    if n_machines == 1:
        # For single machine, makespan is just sum
        return float(processing_times[order, 0].sum())
    
    # Create new order
    job = order[pos_remove]
    new_order = order[:pos_remove] + order[pos_remove+1:]
    if pos_insert > pos_remove:
        pos_insert -= 1
    new_order.insert(pos_insert, job)
    
    # Find the earliest position where the schedule might change
    start_pos = min(pos_remove, pos_insert)
    
    # Compute prefix completions up to start_pos
    prev = np.zeros(n_machines)
    pt = processing_times
    
    for i in range(start_pos):
        j = new_order[i]
        row = pt[j]
        curr = np.empty(n_machines)
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            if curr[m-1] > prev[m]:
                curr[m] = curr[m-1] + row[m]
            else:
                curr[m] = prev[m] + row[m]
        prev = curr
    
    # Continue from start_pos
    for i in range(start_pos, n):
        j = new_order[i]
        row = pt[j]
        curr = np.empty(n_machines)
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            if curr[m-1] > prev[m]:
                curr[m] = curr[m-1] + row[m]
            else:
                curr[m] = prev[m] + row[m]
        prev = curr
        
    return float(prev[-1])


def _neh_heuristic(processing_times, sort_key='sum'):
    """
    Generate an initial permutation using the NEH heuristic.
    sort_key: 'sum', 'avg', 'last', 'first'
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    if sort_key == 'sum':
        job_sums = processing_times.sum(axis=1)
    elif sort_key == 'avg':
        job_sums = processing_times.mean(axis=1)
    elif sort_key == 'last':
        job_sums = processing_times[:, -1]
    elif sort_key == 'first':
        job_sums = processing_times[:, 0]
    else:
        job_sums = processing_times.sum(axis=1)
        
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan_fast(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _local_search(order, processing_times, deadline, best_order, best_makespan):
    """
    Local search using insertion moves with steepest descent strategy.
    Uses incremental makespan evaluation for speed.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return order[:], best_makespan
    
    current_order = order[:]
    current_ms = best_makespan
    
    no_improvement_count = 0
    max_no_improvement = 3
    
    while time.time() < deadline:
        improved = False
        best_delta = 0.0
        best_move = None
        
        # Try all possible insertion moves
        for i in range(n):
            if time.time() > deadline:
                return best_order, best_makespan
            
            job = current_order[i]
            
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                
                # Calculate new makespan
                ms = _makespan_incremental(current_order, processing_times, i, j)
                
                delta = ms - current_ms
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
        
        if best_move is not None and best_delta < 0:
            i, j = best_move
            job = current_order[i]
            new_order = current_order[:i] + current_order[i+1:]
            if j > i:
                j -= 1
            new_order.insert(j, job)
            current_order = new_order
            current_ms += best_delta
            
            if current_ms < best_makespan:
                best_makespan = current_ms
                best_order = current_order[:]
            
            improved = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                break
    
    return best_order, best_makespan


def _simulated_annealing(order, processing_times, deadline, best_order, best_makespan):
    """
    Simulated annealing with insertion moves and adaptive cooling.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return best_order, best_makespan
    
    current_order = order[:]
    current_ms = _makespan_fast(current_order, processing_times)
    
    # Initial temperature based on makespan
    T = max(1.0, current_ms * 0.05)
    T_min = max(0.01, current_ms * 0.0001)
    
    # Adaptive cooling: start with slower cooling, speed up later
    cooling_rate = 0.995
    
    iterations = 0
    max_iterations = 100000
    
    while T > T_min and time.time() < deadline and iterations < max_iterations:
        # Generate a random neighbor
        # Insertion move
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
            
        job = current_order[i]
        remaining = current_order[:i] + current_order[i+1:]
        
        if j > i:
            j -= 1
        j = max(0, min(j, len(remaining)))
        
        new_order = remaining[:j] + [job] + remaining[j:]
        
        new_ms = _makespan_fast(new_order, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0:
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_makespan:
                best_makespan = current_ms
                best_order = current_order[:]
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                current_order = new_order
                current_ms = new_ms
        
        T *= cooling_rate
        iterations += 1
        
        # Adaptive: if no improvement for a while, cool faster
        if iterations % 1000 == 0:
            cooling_rate = max(0.98, cooling_rate * 0.99)
    
    return best_order, best_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145.0
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate multiple initial solutions
    best_order = []
    best_makespan = float('inf')
    
    sort_keys = ['sum', 'avg', 'last', 'first']
    for key in sort_keys:
        if time.time() > deadline - 5:
            break
        order = _neh_heuristic(processing_times, key)
        ms = _makespan_fast(order, processing_times)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order[:]
            
    # 2. Local Search with Steepest Descent
    # Allocate 25% of time to local search
    ls_deadline = start_time + (deadline - start_time) * 0.25
    best_order, best_makespan = _local_search(
        best_order, processing_times, ls_deadline, best_order, best_makespan
    )
    
    # 3. Simulated Annealing
    # Allocate remaining time to SA
    if time.time() < deadline - 2.0:
        best_order, best_makespan = _simulated_annealing(
            best_order, processing_times, deadline, best_order, best_makespan
        )
        
    # 4. Final Polish with local search
    if time.time() < deadline - 1.0:
        best_order, best_makespan = _local_search(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    return best_order

# EVOLVE-BLOCK-END
