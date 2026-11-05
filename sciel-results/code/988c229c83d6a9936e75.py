# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Build an N x N edge weight matrix approximating incremental makespan.
    2. Generate multiple initial solutions (NN, Random, Greedy).
    3. Apply Intensive Local Search (Swap, 2-Opt, Insertion) with time budgeting.
    """
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Helper for fast makespan
    def compute_makespan(seq):
        if not seq:
            return 0.0
        last_comp = np.zeros(n_machines)
        # Use local variables for speed
        p_local = p
        n_m = n_machines
        for j in seq:
            prev = 0.0
            # Unroll loop for small M? No, keep generic.
            for k in range(n_m):
                val = last_comp[k]
                if prev > val:
                    val = prev
                prev = val + p_local[j, k]
                last_comp[k] = prev
        return last_comp[-1]

    # Step 1: Build edge weight matrix W[i][j]
    # W[i][j] = makespan([i, j]) - makespan([i])
    # Precompute sum of p[i] for makespan([i])
    ms_single = p.sum(axis=1)
    
    W = np.zeros((n_jobs, n_jobs))
    for i in range(n_jobs):
        C_i = np.cumsum(p[i])
        for j in range(n_jobs):
            if i == j:
                continue
            # Calculate makespan of [i, j]
            # C_j[k] = max(C_i[k], C_j[k-1]) + p[j,k]
            C_j = np.zeros(n_machines)
            for k in range(n_machines):
                if k == 0:
                    C_j[0] = C_i[0] + p[j, 0]
                else:
                    c_prev = C_j[k-1]
                    c_i_k = C_i[k]
                    if c_prev > c_i_k:
                        C_j[k] = c_prev + p[j, k]
                    else:
                        C_j[k] = c_i_k + p[j, k]
            W[i][j] = C_j[-1] - ms_single[i]

    # Step 2: Initial Solutions
    
    def nn_heuristic(start_job):
        visited = [start_job]
        unvisited = set(range(n_jobs))
        unvisited.remove(start_job)
        current = start_job
        while unvisited:
            # Find best next job
            best_j = -1
            best_w = float('inf')
            w_row = W[current]
            for j in unvisited:
                w_val = w_row[j]
                if w_val < best_w:
                    best_w = w_val
                    best_j = j
            visited.append(best_j)
            unvisited.remove(best_j)
            current = best_j
        return visited

    def random_heuristic():
        seq = list(range(n_jobs))
        random.shuffle(seq)
        return seq

    def greedy_pt_heuristic():
        # Sort by total processing time descending
        total_pt = p.sum(axis=1)
        return list(np.argsort(-total_pt))

    # Generate diverse starts
    starts = []
    
    # 1. NN from job with max total PT
    total_pt = p.sum(axis=1)
    first_job = int(np.argmax(total_pt))
    starts.append(nn_heuristic(first_job))
    
    # 2. NN from job with min total PT
    min_job = int(np.argmin(total_pt))
    starts.append(nn_heuristic(min_job))
    
    # 3. Greedy PT
    starts.append(greedy_pt_heuristic())
    
    # 4. Random
    starts.append(random_heuristic())
    
    # Remove duplicates (optional, but good for efficiency)
    unique_starts = []
    for s in starts:
        if s not in unique_starts:
            unique_starts.append(s)
            
    starts = unique_starts

    # Step 3: Local Search
    # We will use a time-budgeted local search that cycles through neighborhoods.
    
    def local_search(seq, time_limit):
        current_ms = compute_makespan(seq)
        n = len(seq)
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Cycle through neighborhoods: Swap, 2-Opt, Insertion
            
            # 1. Swap (Exchange two jobs)
            # O(N^2) moves
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n):
                    # Swap i and j
                    new_seq = seq.copy()
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    new_ms = compute_makespan(new_seq)
                    if new_ms < current_ms - 1e-9:
                        seq = new_seq
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and time.time() - start_time < time_limit:
                # 2. 2-Opt (Reverse segment)
                for i in range(1, n - 1):
                    if time.time() - start_time > time_limit:
                        break
                    for j in range(i + 1, n):
                        new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                        new_ms = compute_makespan(new_seq)
                        if new_ms < current_ms - 1e-9:
                            seq = new_seq
                            current_ms = new_ms
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved and time.time() - start_time < time_limit:
                # 3. Insertion (Move one job to another position)
                for i in range(1, n):
                    if time.time() - start_time > time_limit:
                        break
                    job = seq[i]
                    base_seq = seq[:i] + seq[i+1:]
                    for pos in range(1, len(base_seq) + 1):
                        if pos == i:
                            continue
                        new_seq = base_seq[:pos] + [job] + base_seq[pos:]
                        new_ms = compute_makespan(new_seq)
                        if new_ms < current_ms - 1e-9:
                            seq = new_seq
                            current_ms = new_ms
                            improved = True
                            break
                    if improved:
                        break
                        
        return seq, current_ms

    # Run local search on each start with split time
    total_time_limit = 110.0 # Leave 10s buffer
    time_per_start = total_time_limit / len(starts)
    
    best_seq = starts[0]
    best_ms = compute_makespan(best_seq)
    
    for i, start_seq in enumerate(starts):
        # Adjust time for remaining starts
        remaining_starts = len(starts) - i
        if remaining_starts > 1:
            time_budget = (total_time_limit - (time.time() - (time.time() - total_time_limit))) / remaining_starts
            # Simpler: just use fixed slice, but check time
            time_budget = time_per_start
        else:
            time_budget = time.time() + total_time_limit - time.time() # Remaining time
            
        # Ensure we don't exceed total limit
        current_time = time.time()
        if current_time + time_budget > time.time() + total_time_limit:
             time_budget = (time.time() + total_time_limit) - current_time
             
        if time_budget <= 0:
            break

        seq_refined, ms_refined = local_search(start_seq, time_budget)
        if ms_refined < best_ms:
            best_seq = seq_refined
            best_ms = ms_refined
            
    return best_seq
# EVOLVE-BLOCK-END
