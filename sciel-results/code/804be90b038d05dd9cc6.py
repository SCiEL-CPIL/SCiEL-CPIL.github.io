import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        # perm is a list of job indices
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
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
    best_ms = calc_makespan(seq)

    # 2. Local Search: Or-opt (insertion) neighborhood
    # Remove job at position i, insert at position j (j > i)
    # This is O(n^2 * n_machines) per pass, much faster than swap 2-opt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(seq)
        # Iterate over all pairs (i, j) with i < j
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Remove job at i, insert at j
                job = seq[i]
                new_seq = seq[:i] + seq[i+1:j+1] + [job] + seq[j+1:]
                ms = calc_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = list(new_seq)
                    seq = new_seq
                    improved = True
    
    # 3. Simulated Annealing to escape local minima
    if time.time() < deadline - 10.0:
        current_seq = list(best_seq)
        current_ms = best_ms
        
        # Temperature schedule: start high, cool down
        t0 = 10.0
        t_end = 0.01
        cooling_rate = 0.999
        
        temp = t0
        no_improve_count = 0
        max_no_improve = 5000
        
        while temp > t_end and time.time() < deadline - 2.0:
            # Generate a random insertion move
            i = random.randint(0, len(current_seq) - 2)
            j = random.randint(i + 1, len(current_seq) - 1)
            
            # Remove job at i, insert at j
            job = current_seq[i]
            new_seq = current_seq[:i] + current_seq[i+1:j+1] + [job] + current_seq[j+1:]
            new_ms = calc_makespan(new_seq)
            
            delta = new_ms - current_ms
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_seq = new_seq
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = list(current_seq)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            temp *= cooling_rate
            
            if no_improve_count > max_no_improve:
                break
    
    return best_seq
