# EVOLVE-BLOCK-START
import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Fix the first job as the one with the highest total processing time.
    2. Build an N x N edge weight matrix approximating incremental makespan.
    3. Use Nearest-Neighbor TSP heuristic and CDS heuristic for initial solutions.
    4. Apply First-Improvement 2-opt, Insertion, and Or-Opt local search.
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
    
    # Step 2: Build edge weight matrix W[i][j]
    # W[i][j] = makespan([i, j]) - makespan([i])
    def makespan_1job(i):
        return p[i, -1]
    
    def makespan_2jobs(i, j):
        # Fast calculation for 2 jobs
        # Machine 0: max(0, 0) + p[i,0] -> p[i,0]
        # Machine k: max(C_{i,k-1}, C_{j,k-1}) + p[j,k]
        # Let's just do the loop, it's small
        last_comp_i = np.zeros(n_machines)
        last_comp_j = np.zeros(n_machines)
        for k in range(n_machines):
            last_comp_i[k] = last_comp_i[k-1] + p[i, k] if k > 0 else p[i, k]
        
        # Actually, makespan of [i] is just sum? No, it's the completion time on last machine.
        # For a single job i, C[i][k] = C[i][k-1] + p[i][k]. So C[i][M-1] = sum(p[i]).
        # Wait, the seed code said `return p[i][-1]`. That is INCORRECT for makespan of a single job in a flow shop?
        # No, for a single job, the completion time on machine k is sum_{l=0}^k p[i][l].
        # The makespan is C[i][M-1] = sum(p[i]).
        # The seed code `return p[i][-1]` is wrong. It returns only the processing time on the last machine.
        # Let's fix this in the new implementation.
        
        # Correct makespan for single job i:
        ms_i = np.sum(p[i])
        
        # Makespan for [i, j]:
        # C[i][k] = C[i][k-1] + p[i][k]
        # C[j][k] = max(C[i][k], C[j][k-1]) + p[j][k]
        # C[i] is just cumulative sum of p[i].
        C_i = np.cumsum(p[i])
        
        C_j = np.zeros(n_machines)
        for k in range(n_machines):
            if k == 0:
                C_j[0] = C_i[0] + p[j, 0]
            else:
                C_j[k] = max(C_i[k], C_j[k-1]) + p[j, k]
        
        return C_j[-1] - ms_i

    W = np.zeros((n_jobs, n_jobs))
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j:
                W[i][j] = makespan_2jobs(i, j)
            else:
                W[i][j] = 0.0

    # Step 3: Initial Solutions
    # Solution A: Nearest Neighbor on W starting from first_job
    def nn_heuristic(start_job):
        visited = [start_job]
        unvisited = set(remaining_jobs)
        current = start_job
        while unvisited:
            best_j = min(unvisited, key=lambda j: W[current][j])
            visited.append(best_j)
            unvisited.remove(best_j)
            current = best_j
        return visited

    # Solution B: CDS Heuristic (Nelson's)
    def cds_heuristic():
        # Nelson's CDS:
        # 1. Compute total processing time for each job
        # 2. Sort jobs by total processing time
        # 3. Construct two sequences:
        #    - Sequence 1: Add jobs in sorted order to the end
        #    - Sequence 2: Add jobs in sorted order to the beginning
        # 4. For each sequence, insert the remaining jobs one by one (in sorted order)
        #    into the position that minimizes makespan.
        
        sorted_jobs = np.argsort(total_pt)
        
        # Seq 1: Start with first job in sorted order, add rest to end
        seq1 = [int(sorted_jobs[0])]
        for j in sorted_jobs[1:]:
            seq1.append(int(j))
            
        # Seq 2: Start with first job in sorted order, add rest to beginning
        seq2 = [int(sorted_jobs[0])]
        for j in sorted_jobs[1:]:
            seq2.insert(0, int(j))
        
        # Now, refine by inserting remaining jobs? 
        # Standard CDS usually starts with the longest job and inserts others.
        # Let's do a simpler CDS variant:
        # Start with the job with max total PT (which is first_job).
        # Then insert the remaining jobs (sorted by PT) into the best position.
        
        def insert_best_position(seq, job):
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                ms = compute_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            return seq[:best_pos] + [job] + seq[best_pos:]

        seq_cds = [first_job]
        # Sort remaining jobs by total PT descending
        rem_sorted = sorted(remaining_jobs, key=lambda x: total_pt[x], reverse=True)
        for j in rem_sorted:
            seq_cds = insert_best_position(seq_cds, j)
            
        return seq_cds

    # Helper for fast makespan
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

    # Step 4: Local Search
    # We will use First Improvement to be fast.
    
    def local_search(seq, time_limit):
        current_ms = compute_makespan(seq)
        improved = True
        start_time = time.time()
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 1. 2-Opt
            n = len(seq)
            for i in range(1, n - 1):
                if time.time() - start_time > time_limit:
                    return seq, current_ms
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
                    
            if not improved:
                # 2. Insertion
                for i in range(1, n):
                    if time.time() - start_time > time_limit:
                        return seq, current_ms
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
                        
            if not improved:
                # 3. Or-Opt (move block of size 2)
                for i in range(1, n - 1):
                    if time.time() - start_time > time_limit:
                        return seq, current_ms
                    block = seq[i:i+2]
                    base_seq = seq[:i] + seq[i+2:]
                    for pos in range(1, len(base_seq) + 1):
                        if pos == i or pos == i + 1:
                            continue
                        new_seq = base_seq[:pos] + block + base_seq[pos:]
                        new_ms = compute_makespan(new_seq)
                        if new_ms < current_ms - 1e-9:
                            seq = new_seq
                            current_ms = new_ms
                            improved = True
                            break
                    if improved:
                        break
                        
        return seq, current_ms

    # Generate initial solutions
    seq_nn = nn_heuristic(first_job)
    seq_cds = cds_heuristic()
    
    # Run local search on both with split time
    total_time_limit = 105.0 # Leave 15s buffer
    time_per_start = total_time_limit / 2
    
    best_seq = seq_nn
    best_ms = compute_makespan(seq_nn)
    
    # Refine NN
    seq_refined, ms_refined = local_search(seq_nn, time_per_start)
    if ms_refined < best_ms:
        best_seq = seq_refined
        best_ms = ms_refined
        
    # Refine CDS
    if time.time() < time.time() + time_per_start: # Just in case
        seq_refined_cds, ms_refined_cds = local_search(seq_cds, time_per_start)
        if ms_refined_cds < best_ms:
            best_seq = seq_refined_cds
            best_ms = ms_refined_cds
            
    return best_seq
# EVOLVE-BLOCK-END
