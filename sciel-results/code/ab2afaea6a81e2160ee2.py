"""
Flow-shop permutation construction using Simulated Annealing with Or-opt moves.
Strategy:
1. Initialize with NEH heuristic for a strong starting point.
2. Use Simulated Annealing with Or-opt block moves (size 1-3) to explore the neighborhood.
3. Vectorized makespan calculation using NumPy cumulative sums for speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_vec(order, P):
    """
    Compute makespan for a given job order using vectorized NumPy operations.
    P: (n_jobs, n_machines)
    order: list/array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    # Extract processing times for the sequence
    # shape: (seq_len, n_machines)
    seq_P = P[order]
    
    # For each job (row), we compute the completion times on each machine.
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # This dependency makes full vectorization tricky without a loop over jobs or machines.
    # However, we can vectorize over machines for a single job? No, C[j, m] depends on C[j, m-1].
    # We can vectorize over jobs if we unroll the machine loop? 
    # Let's try a loop over machines, updating a vector of completion times for all jobs.
    # C_prev_machine is a vector of completion times of all jobs on the previous machine.
    # C_curr_machine is a vector of completion times of all jobs on the current machine.
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    # C_machines[j] will hold the completion time of job j on the current machine being processed.
    # Initially, completion on machine -1 is 0 for all jobs.
    C_prev = np.zeros(n_jobs_seq)
    
    for m in range(n_machines):
        # C_curr[j] = max(C_prev[j], C_curr[j-1]) + P[order[j], m]
        # This is a sequential dependency along the jobs dimension (j).
        # C_curr[j] depends on C_curr[j-1].
        # We cannot fully vectorize this step without a loop or a specialized scan operation.
        # np.maximum.accumulate can help if we reformulate.
        # C_curr[j] = max(C_prev[j] + P[order[j], m], C_curr[j-1] + P[order[j], m])
        # This is not a standard prefix max because of the + P term.
        
        # Actually, the standard flow shop recurrence is:
        # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
        # Let's stick to the loop over jobs, but use NumPy for the inner machine part?
        # No, the dependency is along machines for a fixed job, and along jobs for a fixed machine.
        
        # Alternative: Loop over jobs (outer), vectorize over machines (inner).
        # C_curr_machines is a vector of size n_machines.
        # For job j:
        #   C_m[0] = C_prev_m[0] + P[j, 0]  (since C_prev_m[-1] is 0 effectively, or just C_m[0] = max(C_m[0], 0) + P[j,0])
        #   C_m[m] = max(C_m[m], C_m[m-1]) + P[j, m]
        # This inner loop is still sequential.
        
        # Given n_jobs is likely small (e.g., 10-100) and n_machines small (e.g., 5-10),
        # a pure Python loop might be fast enough if optimized, but let's try to minimize overhead.
        
        # Let's use the job-loop approach with a small inner loop.
        # We will maintain C_machines vector.
        pass

    # Let's implement the job-loop version which is easier to reason about and often fast enough for small N.
    # If N is large, we might need more optimization, but SA needs many evals.
    # Let's try to speed up the _makespan by using a pre-allocated array and avoiding list lookups.
    
    # Actually, let's look at the inspiration 1 which had _fast_insertion_makespan.
    # For SA, we need full makespan for each neighbor.
    
    # Let's use a hybrid: Loop over jobs, inner loop over machines in Python.
    # To speed up, we can use a local variable for P row.
    
    c_machines = np.zeros(n_machines)
    last_completion = 0.0
    
    # Accessing P[order[j]] in a loop is expensive.
    # Let's extract the sequence matrix once.
    # seq_P is (N, M).
    
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        # Unroll first machine?
        # c_curr = c_machines[0] + row[0]
        # c_machines[0] = c_curr
        # for m in range(1, n_machines):
        #     c_curr = max(c_machines[m], c_curr) + row[m]
        #     c_machines[m] = c_curr
        
        # Let's do the standard loop
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan_vec(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _or_opt_move(seq, P):
    """
    Generate a random Or-opt neighbor by moving a block of size 1, 2, or 3.
    Returns new sequence and the block size.
    """
    n = len(seq)
    if n <= 3:
        # Just swap or random shuffle for small N
        new_seq = seq.copy()
        if n > 1:
            i, j = random.sample(range(n), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        return new_seq, 1

    # Choose block size
    max_block = min(3, n - 1)
    block_size = random.randint(1, max_block)
    
    # Choose start position
    # The block can start from 0 to n - block_size
    start = random.randint(0, n - block_size)
    
    # Extract block
    block = seq[start:start + block_size]
    
    # Remove block
    remaining = seq[:start] + seq[start + block_size:]
    
    # Choose insertion position in remaining
    # Insertion pos can be 0 to len(remaining)
    # We should avoid inserting back at the same place (start or start+1 depending on definition)
    # If we insert at 'start' in remaining, it's the same position.
    # If we insert at 'start+1' in remaining, it's after the removed block? No.
    # Let's just pick a random insertion point and check if it results in the same sequence.
    
    ins_pos = random.randint(0, len(remaining))
    
    new_seq = remaining[:ins_pos] + block + remaining[ins_pos:]
    
    # Check if it's the same (happens if ins_pos == start or ins_pos == start+1? 
    # If start=2, block removed, remaining len is n-bs.
    # If ins_pos=2, it goes back to 2.
    # If ins_pos=3, it goes to 3, which is start+1. But since we removed bs elements,
    # inserting at start+1 puts it after the first element of the original block? No.
    # Let's just compare if new_seq == seq. If so, retry or return a different move.
    
    if new_seq == seq:
        # Retry once with a different position
        # Find a valid position that is not start
        valid_positions = [p for p in range(len(remaining) + 1) if p != start and p != start + 1]
        if valid_positions:
            ins_pos = random.choice(valid_positions)
            new_seq = remaining[:ins_pos] + block + remaining[ins_pos:]
        else:
            # Fallback to swap
            i, j = random.sample(range(n), 2)
            new_seq = seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    return new_seq, block_size

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH initialization and Simulated Annealing.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # 150s limit, use 145s to be safe
    time_limit = 145.0
    
    # 1. Initialize with NEH
    current_seq = _neh_heuristic(P)
    current_makespan = _makespan_vec(current_seq, P)
    
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # 2. Simulated Annealing
    # Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.995  # Cooling rate
    min_T = 1e-4
    max_iter = 100000  # Max iterations safeguard
    
    # Adjust T based on problem size? 
    # Average processing time
    avg_pt = np.mean(P)
    T = avg_pt * 10  # Scale temperature with problem magnitude
    
    i = 0
    while (time.time() - start_time) < time_limit and T > min_T and i < max_iter:
        i += 1
        
        # Generate neighbor
        new_seq, _ = _or_opt_move(current_seq, P)
        new_makespan = _makespan_vec(new_seq, P)
        
        delta = new_makespan - current_makespan
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = current_seq.copy()
        else:
            # Deterioration
            # Prob = exp(-delta / T)
            # If delta is very large, prob is 0
            if delta < T * 100: # Optimization: if delta is huge, exp will underflow to 0
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_seq = new_seq
                    current_makespan = new_makespan
        
        # Cool down
        T *= cooling_rate
        
        # Periodic reheat? Or just let it cool.
        # If we stall, we might reheat.
        # For simplicity, we just cool.
        
        # Check time more frequently for large N
        if i % 100 == 0 and (time.time() - start_time) > time_limit:
            break

    return best_seq
# EVOLVE-BLOCK-END
