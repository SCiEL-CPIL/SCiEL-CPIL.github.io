"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Calculate the makespan for a given job order in a flow shop."""
    if not order:
        return 0.0
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    curr = np.zeros(n_machines)
    for j in order:
        curr[0] += processing_times[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], curr[m]) + processing_times[j, m]
    return curr[-1]


def _neh_heuristic(processing_times):
    """Generate initial permutation using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    job_sums = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-job_sums))
    
    current_sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        # Optimization: Only check a few positions around the middle or best fit
        # For NEH, standard is to check all positions. 
        # To speed up, we can use a simplified insertion if n is large, but NEH is O(N^2*M) which is usually fine for N < 1000.
        # Let's stick to standard NEH but optimize the inner loop slightly.
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_sequence.insert(best_pos, job)
    
    return current_sequence


def _adjacent_swap_ls(order, processing_times, deadline):
    """Local search using adjacent swaps (2-opt restricted to adjacent)."""
    n = len(order)
    if n <= 1:
        return order, _makespan(order, processing_times)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            # Swap i and i+1
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_makespan = _makespan(new_order, processing_times)
            
            if new_makespan < current_makespan:
                current_order = new_order
                current_makespan = new_makespan
                improved = True
                # Restart the scan from the beginning (or i-1) to ensure convergence
                # Restarting from 0 is safer but slower. Let's restart from max(0, i-1)
                # Actually, simple full restart is standard for Hill Climbing
                break
    
    return current_order, current_makespan


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
    
    # Phase 1: NEH Heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Phase 2: Adjacent Swap Local Search
    # Allocate most of the time to this
    ls_deadline = deadline - 1.0
    if time.time() < ls_deadline:
        current_order, current_makespan = _adjacent_swap_ls(current_order, processing_times, ls_deadline)
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = current_order[:]
    
    # Phase 3: If time remains, try a few random restarts with NEH + LS
    remaining_time = deadline - time.time()
    if remaining_time > 1.0 and n_jobs > 5:
        import random
        rng = random.Random(42)
        
        # Estimate how many restarts we can do
        # NEH is O(N^2*M), LS is O(N^2*M * iterations)
        # Let's just do a few quick ones
        max_restarts = min(5, int(remaining_time / 2.0))
        
        for _ in range(max_restarts):
            if time.time() > deadline - 0.5:
                break
            
            # Perturb best order
            start_order = best_order[:]
            num_swaps = max(1, n_jobs // 10)
            for _ in range(num_swaps):
                i, j = rng.sample(range(n_jobs), 2)
                start_order[i], start_order[j] = start_order[j], start_order[i]
            
            # Quick NEH on perturbed? No, just LS on perturbed is faster
            # Or just run NEH again? NEH is constructive, so it ignores start_order.
            # Let's use a random start + LS
            random_order = list(range(n_jobs))
            rng.shuffle(random_order)
            
            ls_deadline_restart = time.time() + max(0.1, remaining_time / 2.0)
            rand_order, rand_ms = _adjacent_swap_ls(random_order, processing_times, ls_deadline_restart)
            
            if rand_ms < best_makespan:
                best_makespan = rand_ms
                best_order = rand_order[:]
    
    return best_order

# EVOLVE-BLOCK-END
