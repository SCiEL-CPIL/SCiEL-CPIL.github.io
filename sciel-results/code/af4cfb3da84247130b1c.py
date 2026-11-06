import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        if not perm:
            return 0.0
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
    
    # Fast delta evaluation for insertion moves
    # When moving job from position i to position j (i < j), the jobs between i and j shift left by one
    # We can compute the delta in O(n_machines) instead of O(n_jobs * n_machines)
    def calc_makespan_insertion(perm, i, j):
        """
        Compute makespan after moving job at position i to position j (i < j).
        The jobs at positions i+1..j shift left by one, and the job at i goes to position j.
        """
        if i >= j:
            return calc_makespan(perm)
        
        # Build the new sequence implicitly
        # Positions 0..i-1: same
        # Positions i..j-1: old positions i+1..j
        # Position j: old position i
        # Positions j+1..: same
        
        # We need to recompute makespan from scratch for the affected part
        # But we can use the fact that only positions i..j are affected
        # Actually, for simplicity and correctness, let's just compute full makespan
        # but we can optimize by noting that the prefix before i is unchanged
        
        # For now, use full computation but with early termination if we detect it's worse
        # Actually, let's just do the full computation for correctness
        new_perm = perm[:i] + perm[i+1:j+1] + [perm[i]] + perm[j+1:]
        return calc_makespan(new_perm)
    
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
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        if time.time() > deadline:
            return list(seq)
    
    best_seq = list(seq)
    best_ms = calc_makespan(seq)
    
    # 2. Iterated Local Search with Or-opt moves
    # We'll do multiple passes of insertion neighborhood search, with random restarts
    
    def or_opt_local_search(perm, best_ms_val, time_limit):
        """
        Perform Or-opt local search: remove a job and reinsert it at a different position.
        Returns (best_perm, best_ms).
        """
        current = list(perm)
        current_ms = best_ms_val
        n = len(current)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n - 1):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    job = current[i]
                    new_seq = current[:i] + current[i+1:j+1] + [job] + current[j+1:]
                    ms = calc_makespan(new_seq)
                    if ms < current_ms:
                        current_ms = ms
                        current = new_seq
                        improved = True
        return current, current_ms
    
    # 3. Simulated Annealing
    if time.time() < deadline - 5.0:
        current_seq = list(best_seq)
        current_ms = best_ms
        
        # Adaptive temperature
        t0 = max(5.0, best_ms * 0.01)
        t_end = 0.01
        cooling_rate = 0.9995
        
        temp = t0
        no_improve_count = 0
        max_no_improve = 3000
        
        while temp > t_end and time.time() < deadline - 2.0:
            n = len(current_seq)
            # Generate a random insertion move
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            
            job = current_seq[i]
            new_seq = current_seq[:i] + current_seq[i+1:j+1] + [job] + current_seq[j+1:]
            new_ms = calc_makespan(new_seq)
            
            delta = new_ms - current_ms
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
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
                # Restart with a new random perturbation
                if time.time() > deadline - 2.0:
                    break
                # Perturb the best solution
                for _ in range(min(5, len(best_seq) // 2)):
                    a = random.randint(0, len(best_seq) - 2)
                    b = random.randint(a + 1, len(best_seq) - 1)
                    job = best_seq[a]
                    best_seq = best_seq[:a] + best_seq[a+1:b+1] + [job] + best_seq[b+1:]
                current_seq = list(best_seq)
                current_ms = calc_makespan(current_seq)
                temp = t0 * 0.5
                no_improve_count = 0
    
    # 4. Final local search polish
    if time.time() < deadline - 1.0:
        final_seq, final_ms = or_opt_local_search(best_seq, best_ms, deadline - 1.0)
        if final_ms < best_ms:
            best_ms = final_ms
            best_seq = final_seq
    
    return best_seq
