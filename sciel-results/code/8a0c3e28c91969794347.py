"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, P):
    """
    Calculate the makespan for a given permutation of jobs.
    P: (N, M) array of processing times.
    perm: list/array of job indices.
    """
    if len(perm) == 0:
        return 0.0
    
    # C[i] is the completion time of job perm[i]
    # We track the completion time of the last machine for each job,
    # but since it's a flow shop, we can just track the cumulative time
    # across machines for the sequence.
    
    # Efficient calculation:
    # time_on_machine[m] is the time when machine m becomes free for the next job in the sequence.
    # Actually, for a flow shop:
    # C[j][k] = C[j-1][k] + P[j][k]  (if job j is after j-1 on machine k? No, that's not right)
    
    # Correct recurrence:
    # Let C[j, m] be the completion time of job j on machine m.
    # C[j, 0] = C[j-1, 0] + P[j, 0]
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    
    # We only need the final makespan, which is C[N-1, M-1].
    # We can compute this in O(N*M).
    
    n = len(perm)
    m = P.shape[1]
    
    # prev_row holds C[j-1, :]
    # curr_row holds C[j, :]
    # We can optimize space, but O(N*M) time is fine.
    
    # Initialize
    # C[-1, :] = 0
    prev_row = np.zeros(m)
    
    for job_idx in perm:
        p = P[job_idx]
        curr_row = np.zeros(m)
        for k in range(m):
            if k == 0:
                curr_row[k] = prev_row[k] + p[k]
            else:
                curr_row[k] = max(prev_row[k], curr_row[k-1]) + p[k]
        prev_row = curr_row
        
    return prev_row[m-1]

def _neh(P):
    """
    Generate an initial permutation using the NEH heuristic.
    """
    n, m = P.shape
    # Sort jobs by total processing time descending
    totals = P.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for job in sorted_jobs:
        # Try inserting this job into all positions in the current perm
        best_pos = 0
        best_ms = float('inf')
        
        # Current length is len(perm)
        for pos in range(len(perm) + 1):
            # Create a new perm with job inserted at pos
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(new_perm, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
        
    return perm

def _local_search(perm, P, deadline=None):
    """
    Improve the permutation using Insertion, Or-opt, and Swap local search.
    Returns the best permutation found.
    """
    current_perm = list(perm)
    current_ms = _makespan(current_perm, P)
    n = len(current_perm)
    
    # If N is very small, we can be exhaustive. If large, we need to be careful.
    # We'll use a loop that checks for improvements.
    
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
            
        improved = False
        
        # 1. Insertion Move: Move job i to position j
        # This is O(N^3 * M) per full pass.
        best_move = None
        best_new_ms = current_ms
        
        for i in range(n):
            job = current_perm[i]
            for j in range(n):
                if i == j or i == j - 1:
                    continue
                # Move job i to j
                # If j > i, shifting logic is different, but list slicing handles it
                new_perm = current_perm[:i] + current_perm[i+1:]
                # Insert at j in the new list.
                # Note: indices in new_perm are shifted if j > i.
                # If we remove i, the list shrinks.
                # If j > i, the target index in the shrunk list is j-1.
                # If j <= i, the target index is j.
                
                # Let's just construct it carefully to avoid index errors.
                # Simpler:
                new_perm = list(current_perm)
                new_perm.pop(i)
                # Determine insertion index in the modified list
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                    
                new_perm.insert(insert_idx, job)
                
                ms = _makespan(new_perm, P)
                if ms < best_new_ms - 1e-9:
                    best_new_ms = ms
                    best_move = new_perm
        
        if best_move:
            current_perm = best_move
            current_ms = best_new_ms
            improved = True
            if deadline and time.time() > deadline:
                break
                
        # 2. Or-opt Move: Move a block of size 1, 2, or 3
        # This is more complex. We might skip it if Insertion is already doing well,
        # but Or-opt can escape local optima that single-job insertion can't.
        
        # Let's try Or-opt for blocks of size 2 and 3.
        # We'll iterate over start index and block size.
        
        for block_size in [2, 3]:
            if deadline and time.time() > deadline:
                break
            for i in range(n - block_size + 1):
                block = current_perm[i:i+block_size]
                # Remove block
                base_perm = current_perm[:i] + current_perm[i+block_size:]
                # Try inserting block at all positions
                for j in range(len(base_perm) + 1):
                    # Skip positions that are effectively the same or adjacent to original
                    # Original start i. In base_perm, the gap is at i.
                    # Inserting at i or i+1 puts it back or next to it.
                    if j == i or j == i + 1:
                        continue
                        
                    new_perm = base_perm[:j] + block + base_perm[j:]
                    ms = _makespan(new_perm, P)
                    if ms < current_ms - 1e-9:
                        current_perm = new_perm
                        current_ms = ms
                        improved = True
                        # Restart Or-opt loop as indices changed
                        # Breaking out of inner loops is messy, so we just set improved
                        # and let the outer while loop restart.
                        # However, we want to find the *best* move or just *any*?
                        # First improvement is faster. Best improvement is slower but better.
                        # Given the cost, let's stick to First Improvement for Or-opt to save time,
                        # or break out.
                        # Let's break out of Or-opt section to restart while loop.
                        break
                if improved:
                    break
            if improved:
                break
                
        # 3. Swap Move
        if not improved:
            for i in range(n):
                if deadline and time.time() > deadline:
                    break
                for j in range(i+1, n):
                    new_perm = list(current_perm)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    ms = _makespan(new_perm, P)
                    if ms < current_ms - 1e-9:
                        current_perm = new_perm
                        current_ms = ms
                        improved = True
                        break
                if improved:
                    break
                    
    return current_perm

def order_jobs(processing_times):
    P = np.asarray(processing_times)
    n, m = P.shape
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate initial solution with NEH
    best_perm = _neh(P)
    best_ms = _makespan(best_perm, P)
    
    # 2. Local Search on the initial solution
    best_perm = _local_search(best_perm, P, deadline)
    best_ms = _makespan(best_perm, P)
    
    # 3. Iterated Local Search (ILS)
    # While time remains, perturb the best solution and refine it.
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Perturbation: Randomly swap two jobs or insert a job randomly
        # We use a "kick" strength that might increase if stuck, but simple random is okay.
        perturbed_perm = list(best_perm)
        
        # Choose between Swap and Insertion perturbation
        if rng.random() < 0.5:
            # Swap
            i, j = rng.choice(n, size=2, replace=False)
            perturbed_perm[i], perturbed_perm[j] = perturbed_perm[j], perturbed_perm[i]
        else:
            # Insertion: Pick a job and move it to a random position
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i != j:
                job = perturbed_perm.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                perturbed_perm.insert(j, job)
        
        # Local Search on perturbed solution
        # To save time, we can use a "lighter" local search for ILS iterations
        # or just the same one. The same one is safer for quality.
        candidate_perm = _local_search(perturbed_perm, P, deadline)
        candidate_ms = _makespan(candidate_perm, P)
        
        if candidate_ms < best_ms:
            best_ms = candidate_ms
            best_perm = candidate_perm
            
    return best_perm
# EVOLVE-BLOCK-END
