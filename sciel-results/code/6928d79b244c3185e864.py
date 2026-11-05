# EVOLVE-BLOCK-START
import numpy as np
import time


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Fix the first job as the one with the highest total processing time.
    2. Use NEH-inspired greedy construction for initial solution.
    3. Apply efficient 2-opt and insertion local search with incremental makespan updates.
    """
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: Fix the first job (highest total processing time)
    total_pt = p.sum(axis=1)
    first_job = int(np.argmax(total_pt))
    
    remaining_jobs = [j for j in range(n_jobs) if j != first_job]
    m = len(remaining_jobs)
    
    if m == 0:
        return [first_job]
    
    def compute_makespan(seq, p, n_machines):
        """Compute makespan for a job sequence."""
        if not seq:
            return 0.0
        last_comp = np.zeros(n_machines)
        for j in seq:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[j][k]
                last_comp[k] = prev
        return last_comp[-1]
    
    # Step 2: NEH-inspired greedy construction
    # Sort remaining jobs by NEH key: descending sum of processing times
    # For ties, use the job with larger min(p[j][0], p[j][-1])
    rem_arr = np.array(remaining_jobs)
    rem_sums = p[rem_arr].sum(axis=1)
    # NEH key: primarily descending sum, secondarily by max of first/last machine
    nehe_keys = -rem_sums * 1000 + (p[rem_arr][:, 0] + p[rem_arr][:, -1])
    sorted_indices = np.argsort(nehe_keys, kind='stable')
    initial_seq = [first_job] + [int(rem_arr[i]) for i in sorted_indices]
    
    current_makespan = compute_makespan(initial_seq, p, n_machines)
    best_seq = initial_seq[:]
    best_makespan = current_makespan
    
    # Step 3: Local search - 2-opt and insertion moves
    # Use time budget to limit iterations
    time_limit = 110.0  # seconds
    start_time = time.time()
    
    # 2-opt: reverse a segment in the sequence (excluding position 0)
    def try_2opt(seq, current_ms, p, n_machines):
        n = len(seq)
        improved = False
        for i in range(1, n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                # Reverse seq[i:j+1]
                new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                new_ms = compute_makespan(new_seq, p, n_machines)
                if new_ms < current_ms - 1e-9:
                    return new_seq, new_ms, True
        return seq, current_ms, improved
    
    # Insertion: remove a job and reinsert it at a better position
    def try_insertion(seq, current_ms, p, n_machines):
        n = len(seq)
        improved = False
        for i in range(1, n):  # don't move the first job
            if time.time() - start_time > time_limit:
                break
            job = seq[i]
            # Remove job from position i
            base_seq = seq[:i] + seq[i+1:]
            # Try inserting at each position from 1 to len(base_seq)
            for pos in range(1, len(base_seq) + 1):
                if pos == i:  # same position
                    continue
                new_seq = base_seq[:pos] + [job] + base_seq[pos:]
                new_ms = compute_makespan(new_seq, p, n_machines)
                if new_ms < current_ms - 1e-9:
                    return new_seq, new_ms, True
        return seq, current_ms, improved
    
    # Alternate between 2-opt and insertion until no improvement or time runs out
    seq = best_seq[:]
    current_ms = best_makespan
    no_improve_count = 0
    max_no_improve = 5
    
    while no_improve_count < max_no_improve and time.time() - start_time < time_limit:
        # Try 2-opt
        seq, current_ms, imp = try_2opt(seq, current_ms, p, n_machines)
        if imp:
            no_improve_count = 0
            if current_ms < best_makespan - 1e-9:
                best_makespan = current_ms
                best_seq = seq[:]
        else:
            no_improve_count += 1
        
        # Try insertion
        if time.time() - start_time >= time_limit:
            break
        seq, current_ms, imp = try_insertion(seq, current_ms, p, n_machines)
        if imp:
            no_improve_count = 0
            if current_ms < best_makespan - 1e-9:
                best_makespan = current_ms
                best_seq = seq[:]
        else:
            no_improve_count += 1
    
    return best_seq
# EVOLVE-BLOCK-END
