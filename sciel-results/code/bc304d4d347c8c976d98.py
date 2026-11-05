"""
Flow-shop permutation optimization using Or-Opt Local Search and Simulated Annealing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation.
    O(N*M)
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use lists for speed
    prev = [0.0] * n_machines
    n_m = n_machines
    pt_local = pt
    
    for j in perm:
        curr = [0.0] * n_m
        curr[0] = prev[0] + pt_local[j, 0]
        for m in range(1, n_m):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt_local[j, m]
        prev = curr
    return prev[-1]

def _neh_initial(pt):
    """
    NEH heuristic: Sort jobs by total processing time descending.
    Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """
    Palmer's heuristic (WSPT variant).
    Priority index: sum_{m=1}^{M} (M - m + 1) * t_{jm}
    """
    n_jobs, n_machines = pt.shape
    weights = np.arange(n_machines, 0, -1)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _random_initial(n_jobs, seed):
    """Random initial solution."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_jobs).tolist()

def _or_opt_local_search(pt, perm, deadline):
    """
    Local search using Or-Opt neighborhood (move blocks of size 1, 2, 3).
    Uses a first-improvement strategy.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    
    # Max block size to consider. 3 is standard for Or-Opt.
    max_block_size = 3
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # Try moving blocks of size 1, 2, 3
        for block_size in range(1, max_block_size + 1):
            if time.time() > deadline:
                break
            # i is the start index of the block
            for i in range(n_jobs - block_size + 1):
                if time.time() > deadline:
                    break
                block = current_perm[i:i+block_size]
                
                # Try inserting this block at position k
                # k is the index in the new permutation (after removal)
                # We iterate k from 0 to n_jobs - block_size
                # Note: If k is within the range of the removed block, we skip
                # because that results in the same permutation or a trivial shift.
                
                # We remove the block first to get the base sequence
                # But creating a new list every time is O(N).
                # We can simulate the makespan without creating the full list if we were smarter,
                # but for now, we construct the new perm to evaluate.
                
                # Optimization: We can avoid constructing the list if we just check
                # if k is "close" to i, but let's keep it simple first.
                
                # To avoid redundant work, we can skip k values that result in the same sequence.
                # If we remove block [i, i+bs-1], the length becomes N-bs.
                # We insert at k in the range [0, N-bs].
                # If k >= i, the insertion point in the original sequence is k+bs.
                
                for k in range(n_jobs - block_size + 1):
                    # Skip if inserting back at the same place or adjacent (which is same perm)
                    # If k == i, it's the same.
                    # If k == i-1, it's the same (block shifted left by 1? No, removal then insertion).
                    # Let's just compute.
                    
                    # Construct new permutation
                    # Remove block
                    new_perm = current_perm[:i] + current_perm[i+block_size:]
                    
                    # Adjust k if k > i
                    # If k > i, the index in the new perm is k - block_size?
                    # No, we are inserting into new_perm.
                    # The loop k is 0 to N-bs.
                    # So we just insert at k in new_perm.
                    
                    new_perm = new_perm[:k] + block + new_perm[k:]
                    
                    # Quick check: if new_perm == current_perm, skip
                    # This happens if k == i (inserting back at same spot)
                    # or if k == i-1 (inserting just before the removed spot, effectively same relative order?)
                    # Actually, if we remove [i, i+bs] and insert at i, it's the same.
                    # If we insert at i-1, it's the same.
                    # We can skip these to save time.
                    if k == i or k == i - 1:
                        continue

                    new_ms = _makespan(pt, new_perm)
                    if new_ms < current_ms:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break # Restart the scan from the beginning of the block sizes/indices
                if improved:
                    break
            if improved:
                break
                
    return current_perm, current_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with Or-Opt neighborhood (size 1, 2, 3).
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    if T0 is None:
        T0 = max(10.0, current_ms * 0.5)
        
    T = T0
    rng = np.random.RandomState(42)
    
    max_block_size = 3
    
    while time.time() < deadline and T > 1e-6:
        # Pick a random move: remove a block of size 1, 2, or 3 and insert elsewhere
        block_size = rng.randint(1, max_block_size + 1)
        if n_jobs < block_size:
            block_size = n_jobs
            
        i = rng.randint(0, n_jobs - block_size + 1)
        k = rng.randint(0, n_jobs - block_size + 1)
        
        if k == i or k == i - 1:
            continue
            
        block = current_perm[i:i+block_size]
        
        # Construct new permutation
        new_perm = current_perm[:i] + current_perm[i+block_size:]
        # k is in the range of the new perm (length N-bs)
        new_perm = new_perm[:k] + block + new_perm[k:]
        
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multi-start local search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    # 150s limit, leave 5s margin
    deadline = start_time + 145
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. Random starts
    for seed in range(5):
        candidates.append(_random_initial(n_jobs, seed))
        
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Phase 1: Local Search on each candidate
    # Allocate time evenly or use a deadline for each
    # We have ~145s. Let's give 20s to LS and 120s to SA?
    # Or distribute LS time among candidates.
    # Let's say 10s per candidate for LS, max 5 candidates = 50s.
    # Remaining 90s for SA.
    
    ls_deadline_per_cand = 10.0
    
    for cand in candidates:
        if time.time() > deadline - 60.0: # Leave 60s for SA
            break
        cand_deadline = time.time() + ls_deadline_per_cand
        if cand_deadline > deadline - 60.0:
            cand_deadline = deadline - 60.0
            
        perm, ms = _or_opt_local_search(pt, cand, cand_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: Simulated Annealing from the best solution
    if time.time() < deadline - 2.0:
        sa_deadline = deadline - 1.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 3: If time remains, run LS again on the SA result?
    if time.time() < deadline - 5.0:
        ls_deadline = deadline - 2.0
        perm, ms = _or_opt_local_search(pt, best_perm, ls_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    return best_perm

# EVOLVE-BLOCK-END
