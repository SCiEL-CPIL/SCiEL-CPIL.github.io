import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Vectorized makespan calculation using cumulative sums
    # For a given sequence of job indices, compute makespan efficiently
    def calc_makespan(perm):
        # perm is an array-like of job indices
        # Get processing times for this sequence: shape (len(perm), n_machines)
        pt_seq = processing_times[np.asarray(perm)]
        # For each machine m, the completion time is the cumulative max of:
        # (completion of machine m-1 for same job) + (processing time on machine m)
        # We can compute this iteratively over machines using numpy
        # Start with machine 0: just cumulative sum of processing times
        cumsum = np.cumsum(pt_seq, axis=0)
        # For machine 1 onwards:
        # completion[m] = max(completion[m-1] for same job, completion[m] for previous job) + pt[m]
        # This is equivalent to: C_m(j) = max(C_m(j-1), C_{m-1}(j)) + P_m(j)
        # We need to iterate over machines
        for m in range(1, n_machines):
            # C_m(j) = max(C_{m-1}(j), C_m(j-1)) + P_m(j)
            # C_m(j-1) is the previous element in the cumsum array (which holds C_{m-1} at this point)
            # We need to compute new cumsum for machine m
            new_cumsum = np.empty_like(cumsum)
            new_cumsum[0] = cumsum[0] + pt_seq[0, m]
            for j in range(1, len(perm)):
                new_cumsum[j] = max(cumsum[j], new_cumsum[j-1]) + pt_seq[j, m]
            cumsum = new_cumsum
        return cumsum[-1]

    # Even faster: fully vectorized using a different approach
    # C_m(j) = max over k<=j of (sum_{i=0}^{k-1} P_i(m) + sum_{i=k}^{j} P_i(m+1) ... )
    # This is complex. Let's use a simpler O(n*m) numpy approach
    def calc_makespan_fast(perm):
        perm_arr = np.asarray(perm)
        pt_seq = processing_times[perm_arr]
        # For each machine, compute the cumulative completion times
        # Machine 0: cumsum
        # Machine m: C_m(j) = max(C_{m-1}(j), C_m(j-1)) + P_m(j)
        # We can use a loop over machines but vectorize over jobs
        n_seq = len(perm_arr)
        if n_seq == 0:
            return 0.0
        prev_cum = np.cumsum(pt_seq[:, 0])
        for m in range(1, n_machines):
            # C_m(j) = max(C_{m-1}(j), C_m(j-1)) + P_m(j)
            # This is a recurrence: C_m(j) = max(C_{m-1}(j), C_m(j-1)) + P_m(j)
            # C_m(j-1) depends on C_m(j-1), so we need sequential computation
            # But we can use numpy's cumulative maximum trick
            # Actually, let's just do the loop - it's O(n*m) which is fine
            curr_cum = np.empty(n_seq)
            curr_cum[0] = prev_cum[0] + pt_seq[0, m]
            for j in range(1, n_seq):
                curr_cum[j] = max(prev_cum[j], curr_cum[j-1]) + pt_seq[j, m]
            prev_cum = curr_cum
        return prev_cum[-1]

    # 1. NEH Construction
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [int(sorted_jobs[0])]
    
    for j in sorted_jobs[1:]:
        j = int(j)
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan_fast(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        if time.time() > deadline:
            return seq

    best_seq = list(seq)
    best_ms = best_makespan

    # 2. Insertion-based local search
    # Move each job to the best position in the sequence
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(seq)
        for i in range(n):
            if time.time() > deadline:
                break
            job = seq[i]
            # Remove job from position i
            seq_no_job = seq[:i] + seq[i+1:]
            # Try inserting at each position
            best_pos = i
            best_ms_candidate = float('inf')
            for pos in range(len(seq_no_job) + 1):
                new_seq = seq_no_job[:pos] + [job] + seq_no_job[pos:]
                ms = calc_makespan_fast(new_seq)
                if ms < best_ms_candidate:
                    best_ms_candidate = ms
                    best_pos = pos
            
            if best_ms_candidate < best_ms:
                # Apply the best move
                seq = seq_no_job[:best_pos] + [job] + seq_no_job[best_pos:]
                best_ms = best_ms_candidate
                best_seq = list(seq)
                improved = True
    
    return best_seq
