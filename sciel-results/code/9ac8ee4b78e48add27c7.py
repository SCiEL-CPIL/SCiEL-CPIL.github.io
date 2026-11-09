"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(perm, pt):
    """Calculate makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # C[j, m] is the completion time of job j on machine m
    # We can simulate sequentially
    # completion_time[m] for the last job processed on machine m
    comp = np.zeros(n_machines, dtype=np.float64)
    
    for j in perm:
        # For each machine, the start time is max(completion of previous job on this machine, completion of current job on previous machine)
        # Actually, for a specific job j:
        # C[j, 0] = P[j, 0]
        # C[j, k] = max(C[j, k-1], C[j-1, k]) + P[j, k]
        
        # We maintain 'comp' as the completion times of the *last* job processed on each machine.
        # But we need the completion time of the *current* job on the *previous* machine.
        
        # Let's use a variable 'prev_comp' which is the completion time of job j on machine k-1.
        # And 'comp[k]' is the completion time of the previous job (j-1) on machine k.
        
        prev_comp = 0.0
        for k in range(n_machines):
            # Start time on machine k is max(completion of job j on machine k-1, completion of job j-1 on machine k)
            start_time = max(prev_comp, comp[k])
            # Completion time on machine k
            comp[k] = start_time + pt[j, k]
            # Update prev_comp for the next machine
            prev_comp = comp[k]
            
    return comp[-1]

def _neh(pt):
    """Nawaz, Enscore, and Ham heuristic."""
    n_jobs = pt.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    sorted_indices = np.argsort(-totals)
    
    # Start with the first job
    current_seq = [sorted_indices[0]]
    current_ms = _calc_makespan(current_seq, pt)
    
    for j in sorted_indices[1:]:
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting j at every position in current_seq
        # current_seq length is L, so there are L+1 positions (0 to L)
        for pos in range(len(current_seq) + 1):
            # Create new sequence
            new_seq = current_seq[:pos] + [j] + current_seq[pos:]
            ms = _calc_makespan(new_seq, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert at best position
        current_seq.insert(best_pos, j)
        current_ms = best_ms
        
    return current_seq

def _local_search(perm, pt):
    """First-improvement insertion local search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _calc_makespan(perm, pt)
        
    current_ms = _calc_makespan(perm, pt)
    improved = True
    
    while improved:
        improved = False
        for i in range(n_jobs):
            job = perm[i]
            # Remove job
            perm_without = perm[:i] + perm[i+1:]
            
            # Try inserting at every other position
            for k in range(len(perm_without) + 1):
                if k == i:
                    continue
                new_perm = perm_without[:k] + [job] + perm_without[k:]
                new_ms = _calc_makespan(new_perm, pt)
                
                if new_ms < current_ms:
                    # Accept move
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break # First improvement, restart scan
                    
    return perm, current_ms

def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs = pt.shape[0]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # Set a deadline with a safety margin
    start_time = time.time()
    deadline = start_time + 140.0 # 10 seconds margin
    
    # Initialize with NEH
    best_perm = _neh(pt)
    best_ms = _calc_makespan(best_perm, pt)
    
    # Iterated Local Search
    current_perm = best_perm.copy()
    current_ms = best_ms
    
    # Random seed for reproducibility in local runs, but evaluator might run multiple times.
    # We can just use random module.
    
    while time.time() < deadline:
        # Perturbation: Move 2-3 random jobs to random positions
        # This helps escape local minima
        n_moves = random.randint(2, min(3, n_jobs))
        for _ in range(n_moves):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            # Swap or Move? Move is better for insertion neighborhood.
            # Let's do a random insertion move
            job = current_perm[i]
            del current_perm[i]
            # Insert at random position
            k = random.randint(0, len(current_perm))
            current_perm.insert(k, job)
            
        # Local Search
        current_perm, current_ms = _local_search(current_perm, pt)
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm.copy()
            
    return list(best_perm)

# EVOLVE-BLOCK-END
