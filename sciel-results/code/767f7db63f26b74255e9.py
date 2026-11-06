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


def _local_search_first_improvement(order, processing_times, deadline, best_order, best_makespan):
    """
    Local search using insertion moves with first-improvement strategy.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return order[:], best_makespan
    
    current_order = order[:]
    current_ms = best_makespan
    
    while time.time() < deadline:
        improved = False
        
        # Randomize the start position to avoid getting stuck in a specific pattern
        start_idx = random.randint(0, n - 1)
        
        for i in range(n):
            if time.time() > deadline:
                return best_order, best_makespan
            
            pos = (start_idx + i) % n
            job = current_order[pos]
            
            # Try all insertion positions
            best_pos = pos
            best_eval_ms = current_ms
            
            # Optimization: Only check positions that might improve
            # For first improvement, we can break early if we find an improvement
            # But we need to be careful: moving to pos+1 or pos-1 are the most likely improvements.
            
            # Let's just iterate through all positions for simplicity, but break on first improvement
            found_improvement = False
            for j in range(n):
                if j == pos:
                    continue
                
                # Create new sequence
                # To avoid O(N) copy every time, we can use a list and pop/insert
                # But for small N, it's fine.
                
                # Efficiently create new order
                # Remove job at pos
                new_order = current_order[:pos] + current_order[pos+1:]
                # Insert at j (adjust index if j > pos)
                insert_idx = j
                if j > pos:
                    insert_idx = j - 1
                new_order.insert(insert_idx, job)
                
                ms = _makespan_fast(new_order, processing_times)
                
                if ms < best_eval_ms:
                    best_eval_ms = ms
                    best_pos = j
                    # First improvement strategy: accept immediately?
                    # If we accept immediately, we restart the scan.
                    # This is "Steepest Descent" if we wait for best, "First Improvement" if we take first.
                    # Let's take the first improvement found to be fast.
                    if ms < current_ms:
                        current_order = new_order
                        current_ms = ms
                        if current_ms < best_makespan:
                            best_makespan = current_ms
                            best_order = current_order[:]
                        improved = True
                        found_improvement = True
                        break # Break inner loop, restart outer loop? 
                        # Actually, breaking inner loop and continuing outer loop with the new order is standard.
                        # But we changed current_order, so indices might shift.
                        # Let's just break out of the inner loop and restart the outer loop scan from the beginning or continue?
                        # Standard First Improvement: Apply move, restart scan from beginning.
            
            if found_improvement:
                break # Restart scan
        
        if not improved:
            break
            
    return best_order, best_makespan


def _simulated_annealing(order, processing_times, deadline, best_order, best_makespan):
    """
    Simulated annealing with insertion moves.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return best_order, best_makespan
    
    current_order = order[:]
    current_ms = _makespan_fast(current_order, processing_times)
    
    # Initial temperature based on makespan
    T = max(1.0, current_ms * 0.1)
    T_min = max(0.01, current_ms * 0.001)
    cooling_rate = 0.98
    
    # To speed up, we can limit the number of iterations per temperature
    # Or just loop until time runs out
    
    while T > T_min and time.time() < deadline:
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
            
    # 2. Local Search with First Improvement
    # Allocate 30% of time to local search
    ls_deadline = start_time + (deadline - start_time) * 0.3
    best_order, best_makespan = _local_search_first_improvement(
        best_order, processing_times, ls_deadline, best_order, best_makespan
    )
    
    # 3. Simulated Annealing
    # Allocate remaining time to SA
    if time.time() < deadline - 2.0:
        best_order, best_makespan = _simulated_annealing(
            best_order, processing_times, deadline, best_order, best_makespan
        )
        
    # 4. Final Polish
    if time.time() < deadline - 1.0:
        best_order, best_makespan = _local_search_first_improvement(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    return best_order

# EVOLVE-BLOCK-END
