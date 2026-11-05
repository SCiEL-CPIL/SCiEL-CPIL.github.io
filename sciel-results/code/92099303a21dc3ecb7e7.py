# EVOLVE-BLOCK-START
import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Generate multiple initial solutions using NEH (Nawaz-Enscore-Ham) and randomized NEH.
    2. Apply Local Search with Insertion (Or-opt 1), Block Move (Or-opt 2), and Interchange heuristics.
    """
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Helper: Compute Makespan for a sequence
    def compute_makespan(seq):
        if not seq:
            return 0.0
        last_comp = np.zeros(n_machines)
        for j in seq:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[j, k]
                last_comp[k] = prev
        return last_comp[-1]

    # Step 1: Initial Solutions
    
    # NEH Heuristic
    total_pt = p.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    
    def neh_heuristic():
        seq = [int(sorted_jobs[0])]
        for idx in range(1, n_jobs):
            j = int(sorted_jobs[idx])
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [j] + seq[pos:]
                ms = compute_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [j] + seq[best_pos:]
        return seq

    # Randomized NEH
    def randomized_neh(num_trials=10):
        best_seq = None
        best_ms = float('inf')
        for _ in range(num_trials):
            noise = np.random.uniform(0, total_pt.max() * 0.1, n_jobs)
            noisy_pt = total_pt + noise
            r_sorted_jobs = np.argsort(-noisy_pt)
            
            seq = [int(r_sorted_jobs[0])]
            for idx in range(1, n_jobs):
                j = int(r_sorted_jobs[idx])
                best_pos = 0
                best_ms_local = float('inf')
                for pos in range(len(seq) + 1):
                    new_seq = seq[:pos] + [j] + seq[pos:]
                    ms = compute_makespan(new_seq)
                    if ms < best_ms_local:
                        best_ms_local = ms
                        best_pos = pos
                seq = seq[:best_pos] + [j] + seq[best_pos:]
            
            ms = compute_makespan(seq)
            if ms < best_ms:
                best_ms = ms
                best_seq = seq
        return best_seq

    # Generate initial candidates
    candidates = []
    
    seq_neh = neh_heuristic()
    candidates.append(seq_neh)
    
    seq_rneh = randomized_neh(10)
    candidates.append(seq_rneh)
    
    # Select best candidate
    best_seq = None
    best_ms = float('inf')
    for seq in candidates:
        ms = compute_makespan(seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq

    # Step 2: Local Search
    start_time = time.time()
    time_limit = 105.0 # Leave 15s buffer
    
    current_seq = list(best_seq)
    current_ms = best_ms
    improved = True
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        n = len(current_seq)
        
        best_move_ms = current_ms
        best_move_type = None
        best_move_params = None
        
        # 1. Or-opt 1 (Single Job Insertion)
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            job = current_seq[i]
            base_seq = current_seq[:i] + current_seq[i+1:]
            for pos in range(len(base_seq) + 1):
                if pos == i:
                    continue
                new_seq = base_seq[:pos] + [job] + base_seq[pos:]
                new_ms = compute_makespan(new_seq)
                if new_ms < best_move_ms - 1e-9:
                    best_move_ms = new_ms
                    best_move_type = 'insert1'
                    best_move_params = (i, pos)
        
        # 2. Or-opt 2 (Block of 2)
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            block = current_seq[i:i+2]
            base_seq = current_seq[:i] + current_seq[i+2:]
            for pos in range(len(base_seq) + 1):
                if pos == i or pos == i + 1:
                    continue
                new_seq = base_seq[:pos] + block + base_seq[pos:]
                new_ms = compute_makespan(new_seq)
                if new_ms < best_move_ms - 1e-9:
                    best_move_ms = new_ms
                    best_move_type = 'insert2'
                    best_move_params = (i, pos)
        
        # 3. Interchange Move
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                new_seq = current_seq.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_ms = compute_makespan(new_seq)
                if new_ms < best_move_ms - 1e-9:
                    best_move_ms = new_ms
                    best_move_type = 'swap'
                    best_move_params = (i, j)

        if best_move_type is not None:
            i, pos = best_move_params
            if best_move_type == 'insert1':
                job = current_seq[i]
                base_seq = current_seq[:i] + current_seq[i+1:]
                current_seq = base_seq[:pos] + [job] + base_seq[pos:]
            elif best_move_type == 'insert2':
                block = current_seq[i:i+2]
                base_seq = current_seq[:i] + current_seq[i+2:]
                current_seq = base_seq[:pos] + block + base_seq[pos:]
            elif best_move_type == 'swap':
                current_seq[i], current_seq[pos] = current_seq[pos], current_seq[i]
            
            current_ms = best_move_ms
            improved = True
            continue # Restart with new state

    return current_seq
# EVOLVE-BLOCK-END
