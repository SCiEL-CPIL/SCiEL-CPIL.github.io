import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH heuristic followed by 2-opt local search.
    """
    # Ensure processing_times is a numpy array for efficient indexing
    if not isinstance(processing_times, np.ndarray):
        processing_times = np.array(processing_times)
    
    n_jobs, n_machines = processing_times.shape
    
    # Edge cases
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 150s limit, keep 5s margin
    
    def calc_makespan(perm):
        """
        Calculate makespan for a given permutation.
        perm: list of job indices
        """
        if len(perm) == 0:
            return 0
        # C[j, k] is completion time of job perm[j] on machine k
        # We only need the previous row and current row to save memory, 
        # but for simplicity and speed in Python, we can just track the last completion times per machine.
        # Actually, we need C[j-1, k] and C[j, k-1].
        # Let's use a 1D array for the previous job's completion times on each machine.
        # prev_c[k] = completion time of previous job on machine k
        # curr_c[k] = completion time of current job on machine k
        
        prev_c = [0.0] * n_machines
        # First job
        curr_job = perm[0]
        curr_c = [0.0] * n_machines
        for k in range(n_machines):
            curr_c[k] = curr_c[k-1] + processing_times[curr_job, k] if k > 0 else processing_times[curr_job, k]
            # Wait, the recurrence is C[j,k] = max(C[j-1,k], C[j,k-1]) + p[j,k]
            # For the first job (j=0), C[-1, k] is 0.
            # So C[0, 0] = p[0,0]
            # C[0, 1] = max(0, C[0,0]) + p[0,1] = C[0,0] + p[0,1]
            # So for the first job, it's just cumulative sum.
        
        # Let's re-write the loop for all jobs to be generic and correct
        # prev_c holds C[j-1, :]
        # We need to compute C[j, :]
        
        # Initialize for job 0
        prev_c = [0.0] * n_machines
        current_job = perm[0]
        curr_c = [0.0] * n_machines
        for k in range(n_machines):
            # C[0, k] = max(C[-1, k], C[0, k-1]) + p[0, k]
            # C[-1, k] is 0.
            # C[0, k-1] is curr_c[k-1] (or 0 if k=0)
            if k == 0:
                curr_c[k] = processing_times[current_job, k]
            else:
                curr_c[k] = max(prev_c[k], curr_c[k-1]) + processing_times[current_job, k]
        
        for j in range(1, n_jobs):
            current_job = perm[j]
            # prev_c is C[j-1, :]
            # We compute curr_c which is C[j, :]
            # C[j, 0] = max(C[j-1, 0], 0) + p[j, 0] = C[j-1, 0] + p[j, 0]
            curr_c[0] = prev_c[0] + processing_times[current_job, 0]
            for k in range(1, n_machines):
                # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
                # C[j-1, k] is prev_c[k]
                # C[j, k-1] is curr_c[k-1]
                curr_c[k] = max(prev_c[k], curr_c[k-1]) + processing_times[current_job, k]
            # Swap
            prev_c, curr_c = curr_c, prev_c
        
        # The makespan is the completion time of the last job on the last machine
        # After the loop, prev_c holds C[n_jobs-1, :]
        return prev_c[n_machines - 1]

    def neh_heuristic(pt):
        """
        NEH heuristic: sort by total processing time descending, then insert.
        """
        # Sort jobs by total processing time descending
        totals = pt.sum(axis=1)
        sorted_jobs = np.argsort(-totals)
        
        seq = [sorted_jobs[0]]
        current_makespan = calc_makespan(seq)
        
        for i in range(1, n_jobs):
            job = sorted_jobs[i]
            best_pos = 0
            best_ms = float('inf')
            
            # Try inserting job at each position in current sequence
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                # Optimization: We don't need to recalculate the entire makespan from scratch
                # if we can do incremental, but for simplicity and correctness first:
                ms = calc_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            seq.insert(best_pos, job)
            
            # Check time
            if time.time() > deadline:
                break
                
        return seq

    def two_opt_search(pt, seq):
        """
        2-opt local search: reverse segments of the permutation.
        """
        current_seq = seq[:]
        current_ms = calc_makespan(current_seq)
        improved = True
        
        while improved:
            improved = False
            n = len(current_seq)
            for i in range(n - 1):
                if time.time() > deadline:
                    return current_seq
                for j in range(i + 1, n):
                    # Reverse segment i to j
                    new_seq = current_seq[:i] + current_seq[i:j+1][::-1] + current_seq[j+1:]
                    new_ms = calc_makespan(new_seq)
                    
                    if new_ms < current_ms - 1e-9:
                        current_seq = new_seq
                        current_ms = new_ms
                        improved = True
                        # Restart the search from the beginning or continue?
                        # Restarting is safer to ensure we find the local optimum.
                        # But it's expensive. Let's just break and restart the outer loop.
                        break
                if improved:
                    break # Break inner loop, outer loop will restart
        
        return current_seq

    # 1. Generate initial solution using NEH
    initial_seq = neh_heuristic(processing_times)
    
    # 2. Apply 2-opt local search
    best_seq = two_opt_search(processing_times, initial_seq)
    
    # 3. If time permits, do a few random restarts
    # Random restarts can help escape local optima
    # We will do this only if we have significant time left (> 20s)
    while time.time() < deadline - 20:
        # Generate a random permutation or a perturbed NEH
        # Let's try a random permutation to diversify
        rand_seq = list(np.random.permutation(n_jobs))
        
        # Apply 2-opt
        cand_seq = two_opt_search(processing_times, rand_seq)
        
        # Compare with best
        cand_ms = calc_makespan(cand_seq)
        best_ms = calc_makespan(best_seq)
        
        if cand_ms < best_ms:
            best_seq = cand_seq
            best_ms = cand_ms
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_seq
