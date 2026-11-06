import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # Safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    # Precompute job processing times as a list of lists for faster access in inner loops
    # processing_times is (n_jobs, n_machines)
    pt = processing_times.tolist()
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    # Optimized for speed using local variables
    def calc_makespan(perm):
        # perm is a list of job indices
        completion = [0.0] * n_machines
        for j in perm:
            jt = pt[j]
            prev = 0.0
            for m in range(n_machines):
                # Completion time on machine m for job j
                # Start time is max(completion[m], completion[m-1])
                # completion[m] holds the completion time of the previous job on machine m
                # prev holds the completion time of the current job on machine m-1
                if m == 0:
                    st = completion[0]
                else:
                    st = completion[m]
                    if prev > st:
                        st = prev
                completion[m] = st + jt[m]
                prev = completion[m]
        return completion[-1]

    # 1. NEH Construction
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times).tolist()
    
    seq = [sorted_jobs[0]]
    
    # Helper to find best insertion position for a job in a sequence
    # Returns (best_pos, best_makespan)
    def best_insertion_pos(job, current_seq):
        best_pos = 0
        best_ms = float('inf')
        # We can optimize this by noticing that inserting at pos k affects
        # the schedule from k onwards. However, for simplicity and robustness
        # within the time limit, we calculate full makespan for each candidate.
        # Since n is likely small (typical for these tasks), O(n^2 * m) is fine.
        for pos in range(len(current_seq) + 1):
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        return best_pos, best_ms

    for j in sorted_jobs[1:]:
        best_pos, best_ms = best_insertion_pos(j, seq)
        seq.insert(best_pos, j)
        
        # Time check
        if time.time() > deadline:
            return seq

    best_seq = list(seq)
    best_ms = calc_makespan(best_seq)
    current_seq = list(best_seq)
    current_ms = best_ms

    # 2. Local Search: Or-opt (Insertion) Neighborhood
    # We remove a job (or block) and re-insert it elsewhere.
    # For simplicity and speed, we start with single job insertion.
    
    def local_search_or_opt(seq_in, ms_in):
        seq = list(seq_in)
        ms = ms_in
        n = len(seq)
        improved = True
        # Limit passes to avoid infinite loops if floating point issues arise, 
        # though makespan is integer-like usually.
        max_passes = 100 
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            # Try removing each job and re-inserting it
            for i in range(n):
                if time.time() > deadline:
                    return seq, ms
                job_to_move = seq[i]
                # Remove job
                seq.pop(i)
                
                # Find best insertion position in the shortened sequence
                best_pos, best_new_ms = best_insertion_pos(job_to_move, seq)
                
                # Insert job at best position
                seq.insert(best_pos, job_to_move)
                
                if best_new_ms < ms - 1e-9:
                    ms = best_new_ms
                    improved = True
                    # Note: We do not restart the loop immediately to save time,
                    # but we update the state. The next iteration of the outer
                    # while loop will check for more improvements.
                    # Actually, restarting the 'i' loop is safer for convergence,
                    # but continuing is faster. Let's continue but ensure we
                    # re-evaluate the new position in subsequent 'i' steps.
                    # Since we modified seq, the indices shift.
                    # To be safe and correct, breaking and restarting is better
                    # for strict hill climbing, but might be slow.
                    # Let's just break and restart the pass to handle index shifts correctly.
                    break
            
        return seq, ms

    # 3. Iterated Local Search (ILS)
    # Diversify with perturbations to escape local minima.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        # Local search from current state
        current_seq, current_ms = local_search_or_opt(current_seq, current_ms)
        
        if current_ms < best_ms - 1e-9:
            best_ms = current_ms
            best_seq = list(current_seq)
        
        # Check if we should stop
        if time.time() > deadline - 5.0:
            break
            
        # Perturbation
        # Strategy: Move a random job to a random position (Or-opt 1)
        # Or swap two random jobs.
        # Let's use a stronger perturbation: Remove 2-3 consecutive jobs and re-insert.
        # For simplicity and speed, we use random insertion of 1-2 jobs.
        
        n = len(current_seq)
        if n <= 1:
            break
            
        # Perturb size: 1 or 2
        perturb_size = 1 if n < 10 else 2
        
        for _ in range(perturb_size):
            # Pick a random job to move
            idx = rng.integers(0, n)
            job = current_seq[idx]
            current_seq.pop(idx)
            
            # Pick a random insertion position
            new_idx = rng.integers(0, n)
            current_seq.insert(new_idx, job)
            
        # Recalculate makespan for perturbed solution
        current_ms = calc_makespan(current_seq)
        
        # If the perturbation made it much worse, we might consider reverting
        # to the best known solution to restart, but standard ILS continues
        # from the perturbed state. However, if it's very bad, restarting
        # from best is often better.
        # Let's restart from best if the perturbation is worse than best.
        if current_ms > best_ms:
            current_seq = list(best_seq)
            current_ms = best_ms
            # Apply a smaller perturbation to current_seq to avoid getting stuck
            # in the same local optimum immediately.
            idx = rng.integers(0, n)
            job = current_seq[idx]
            current_seq.pop(idx)
            new_idx = rng.integers(0, n)
            current_seq.insert(new_idx, job)
            current_ms = calc_makespan(current_seq)

    # Final Polish: One more local search pass on the best solution found
    if time.time() < deadline - 1.0:
        final_seq, final_ms = local_search_or_opt(best_seq, best_ms)
        if final_ms < best_ms:
            best_seq = final_seq
            best_ms = final_ms

    return best_seq
