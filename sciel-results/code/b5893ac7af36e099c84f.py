import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    # Optimized: pre-fetch row references to reduce indexing overhead
    def calc_makespan(perm):
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = completion[m]
                    prev = completion[m-1]
                    if prev > st:
                        st = prev
                completion[m] = st + job_times[m]
        return completion[-1]

    # 1. NEH Construction
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [sorted_jobs[0]]
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each position in the current sequence
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        # Time check
        if time.time() > deadline:
            return list(seq)

    best_seq = list(seq)
    best_ms = calc_makespan(best_seq)
    current_seq = list(best_seq)
    current_ms = best_ms

    # 2. Local Search: Hybrid approach
    # Start with 2-opt (any two swaps) to escape local minima quickly,
    # then switch to adjacent swaps for fine-tuning if time permits.
    # We alternate or just run 2-opt until no improvement, then adjacent.
    
    improved = True
    use_2opt = True
    
    # Limit 2-opt iterations to avoid getting stuck in long loops with small gains
    # If n is large, 2-opt is O(n^2) per pass.
    # We'll run full passes until no improvement or time is up.
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_seq)
        
        if use_2opt:
            # 2-Opt: Swap any two positions
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    # Swap
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                    ms = calc_makespan(current_seq)
                    if ms < current_ms:
                        current_ms = ms
                        if ms < best_ms:
                            best_ms = ms
                            best_seq = list(current_seq)
                        improved = True
                    else:
                        # Revert
                        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                
                if time.time() > deadline:
                    break
            
            # If 2-opt found no improvement in a full pass, switch to adjacent swaps
            if not improved:
                use_2opt = False
        else:
            # Adjacent Swaps: Swap i and i+1
            # This is faster and good for fine-tuning
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                # Swap
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                ms = calc_makespan(current_seq)
                if ms < current_ms:
                    current_ms = ms
                    if ms < best_ms:
                        best_ms = ms
                        best_seq = list(current_seq)
                    improved = True
                    # If we improved, restart the scan from the beginning? 
                    # Or just continue? Standard hill climbing usually restarts or continues.
                    # Continuing is faster. Restarting is more thorough.
                    # Let's restart to ensure we don't miss a better sequence formed by the shift.
                    # Actually, for simple hill climbing, continuing is often sufficient and faster.
                    # But restarting guarantees finding the local optimum of the adjacent neighborhood.
                    # Given time constraints, let's just continue. If it improves again, we update.
                    # Wait, if I swap i, i+1 and improve, the neighbors change.
                    # A full pass with restart-on-improve is safer for quality.
                    # Let's restart the loop if we improve.
                    break
                else:
                    # Revert
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
            
            # If no improvement in full pass of adjacent swaps, we are at a local optimum for this neighborhood.
            # We could try random restarts or perturbations, but let's keep it simple.
            # If we are stuck, we might try a random perturbation (SA-like) if time remains.
            # For now, if no improvement, stop.
            if not improved:
                break

    return best_seq
