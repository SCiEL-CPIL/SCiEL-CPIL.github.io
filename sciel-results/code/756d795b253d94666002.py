"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    Strategy: 
    1. Generate multiple initial permutations using different heuristics (NEH, LPT, SPT, Palmer).
    2. Evaluate makespan for each.
    3. Apply local search (2-opt / adjacent swap) to the best initial solution.
    4. Use time budget to perform multiple restarts or deeper search if time permits.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    if n_jobs == 2:
        # Brute force for 2 jobs
        p0 = processing_times[0].sum()
        p1 = processing_times[1].sum()
        # Actually need to check specific makespans
        # Makespan for [0, 1]:
        c0_0 = processing_times[0, 0]
        c1_0 = c0_0 + processing_times[1, 0]
        c1_1 = max(c1_0, processing_times[1, 1]) + processing_times[1, 1] # Wait, max(finish_prev_m, finish_prev_job_on_m-1)
        
        # Let's just use a general makespan calculator
        def calc_makespan(seq):
            if not seq:
                return 0
            # Vectorized-ish calculation
            # C[j, m] = max(C[j, m-1], C[j-1, m]) + P[j, m]
            # We can do this efficiently
            # Initialize C for first job
            C = np.zeros((n_jobs, n_machines))
            for j in range(n_jobs):
                job = seq[j]
                if j == 0:
                    C[0, 0] = processing_times[job, 0]
                    for m in range(1, n_machines):
                        C[0, m] = C[0, m-1] + processing_times[job, m]
                else:
                    C[j, 0] = C[j-1, 0] + processing_times[job, 0]
                    for m in range(1, n_machines):
                        C[j, m] = max(C[j, m-1], C[j-1, m]) + processing_times[job, m]
            return C[-1, -1]

        ms_01 = calc_makespan([0, 1])
        ms_10 = calc_makespan([1, 0])
        return [0, 1] if ms_01 <= ms_10 else [1, 0]

    # If very large, use a fast heuristic to ensure we return *something* valid quickly
    # But local search might be too slow. Let's check size.
    # n_jobs * n_machines < 5000 is generally fast for local search in Python if optimized.
    # Let's set a threshold for "large" problems where we skip local search.
    is_large = n_jobs > 100 or (n_jobs * n_machines > 2000)

    def get_initial_solutions():
        sols = []
        # 1. LPT (Longest Processing Time)
        sums = processing_times.sum(axis=1)
        sols.append(np.argsort(-sums).tolist())
        
        # 2. SPT (Shortest Processing Time)
        sols.append(np.argsort(sums).tolist())
        
        # 3. NEH (Nawaz, Enscore, Ham)
        # Sort jobs by total processing time descending
        order = np.argsort(-sums)
        neh_seq = []
        for job in order:
            best_pos = 0
            best_ms = float('inf')
            # Try inserting job at every position
            for pos in range(len(neh_seq) + 1):
                trial = neh_seq[:pos] + [job] + neh_seq[pos:]
                ms = calc_makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            neh_seq = neh_seq[:best_pos] + [job] + neh_seq[best_pos:]
        sols.append(neh_seq)
        
        # 4. Palmer's Index
        # J_k = (n - k + 0.5) * p_k - sum_{i=1}^{k} p_i
        # Sort by J_k descending
        # This is a bit more complex to implement exactly, let's use a simpler heuristic:
        # Sort by average processing time
        avg = processing_times.mean(axis=1)
        sols.append(np.argsort(-avg).tolist())
        
        # 5. Random (a few)
        for _ in range(3):
            rand_seq = list(range(n_jobs))
            np.random.shuffle(rand_seq)
            sols.append(rand_seq)
            
        return sols

    def local_search(seq, time_limit):
        current_seq = seq[:]
        current_ms = calc_makespan(current_seq)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(current_seq)
            # Adjacent swap
            for i in range(n - 1):
                if time.time() > time_limit:
                    break
                # Swap i and i+1
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                new_ms = calc_makespan(current_seq)
                if new_ms < current_ms:
                    current_ms = new_ms
                    improved = True
                else:
                    # Swap back
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
            
            # 2-opt (reverse segment) - only if small enough
            if n < 50:
                for i in range(n - 1):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, n):
                        if time.time() > time_limit:
                            break
                        # Reverse segment i..j
                        current_seq[i:j+1] = current_seq[i:j+1][::-1]
                        new_ms = calc_makespan(current_seq)
                        if new_ms < current_ms:
                            current_ms = new_ms
                            improved = True
                        else:
                            current_seq[i:j+1] = current_seq[i:j+1][::-1]
        
        return current_seq, current_ms

    # 1. Generate initial solutions
    initial_sols = get_initial_solutions()
    
    best_seq = None
    best_ms = float('inf')
    
    # Evaluate all initial solutions
    for sol in initial_sols:
        ms = calc_makespan(sol)
        if ms < best_ms:
            best_ms = ms
            best_seq = sol[:]
            
    # Check time
    if time.time() > deadline - 5:
        return best_seq
        
    # 2. Local Search on the best initial solution
    if not is_large:
        # Run local search
        time_limit_ls = deadline - 2
        best_seq, best_ms = local_search(best_seq, time_limit_ls)
        
        # If time remains, try local search on other good initial solutions
        # Sort initial solutions by makespan
        scored_sols = [(calc_makespan(s), s) for s in initial_sols]
        scored_sols.sort(key=lambda x: x[0])
        
        for ms_init, sol in scored_sols:
            if time.time() > deadline - 5:
                break
            # If this initial solution is already worse than current best, and we are not doing deep search, skip?
            # Actually, different basins might lead to better local optima.
            if ms_init > best_ms * 1.1: # Skip if significantly worse
                continue
                
            time_limit_ls2 = deadline - 2
            # We need to be careful with time allocation.
            # Let's just do one more LS if time permits.
            if time.time() < deadline - 10:
                new_seq, new_ms = local_search(sol, time_limit_ls2)
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_seq = new_seq
            else:
                break

    return best_seq
# EVOLVE-BLOCK-END
