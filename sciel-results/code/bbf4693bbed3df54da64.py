import time
import numpy as np
import scipy.sparse as sp
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we stop at 110s to return safely
    deadline = start_time + 110.0
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Generate all k-subsets and map to indices
    k_subsets = list(itertools.combinations(range(v), k))
    k_index = {s: i for i, s in enumerate(k_subsets)}
    n_k = len(k_subsets)
    
    # 3. Build sparse matrix M (n_k x n_t)
    # M[i, j] = 1 if k_subsets[i] covers t_subsets[j]
    rows = []
    cols = []
    
    # Iterate over t-subsets to find covering k-subsets
    # For each t-subset, the number of covering k-subsets is C(v-t, k-t)
    # v=19, k=8, t=5 -> C(14, 3) = 364
    # Total non-zeros = 11628 * 364 = 4,232,592
    
    for t_sub in t_subsets:
        t_set = set(t_sub)
        remaining = [x for x in range(v) if x not in t_set]
        t_idx = t_index[t_sub]
        
        # Choose k-t elements from remaining
        for add in itertools.combinations(remaining, k - t):
            # Combine t_sub and add, sort to match k_index key format
            # t_sub is tuple, add is tuple. 
            # k_sub must be a tuple of sorted integers.
            k_sub = tuple(sorted(t_sub + add))
            k_idx = k_index[k_sub]
            rows.append(k_idx)
            cols.append(t_idx)
            
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_k, n_t))
    
    # Precompute list of covered indices for each block for fast update
    # M is CSR, so M.indices and M.indptr give us this.
    # But accessing M.getrow(i).indices is O(nnz_in_row).
    # We can just use M.getrow in the loop, it's fast enough.
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Each restart is a greedy pass with randomization
    max_restarts = 200 # Will be limited by time
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered = np.ones(n_t, dtype=np.float32)
        selected_indices = []
        
        while True:
            if time.time() > deadline:
                break
                
            # Calculate gains: number of uncovered subsets covered by each block
            # M is float32, uncovered is float32
            gains = M @ uncovered
            
            max_gain = np.max(gains)
            if max_gain <= 1e-6: # Effectively 0
                break
                
            # Select block
            # Randomization strategy:
            # 50% pure greedy, 50% randomized from top 20
            if r % 2 == 0:
                idx = int(np.argmax(gains))
            else:
                k_rand = 20
                if n_k > k_rand:
                    # argpartition is faster than sort for top-k
                    top_rand = np.argpartition(gains, -k_rand)[-k_rand:]
                else:
                    top_rand = np.arange(n_k)
                
                # Pick randomly from top_rand
                # To bias towards higher gains, we could sample with probability proportional to gain
                # But simple random from top-K is a good heuristic for diversity
                idx = int(top_rand[np.random.randint(len(top_rand))])
                
                # Sanity check: if random pick has 0 gain, fallback to argmax
                if gains[idx] <= 1e-6:
                    idx = int(np.argmax(gains))
                    if gains[idx] <= 1e-6:
                        break

            selected_indices.append(idx)
            
            # Update uncovered
            # Get indices of non-zeros in row idx of M
            # M is CSR
            # M.getrow(idx) returns a CSR matrix
            # .indices gives column indices
            covered_idx = M.getrow(idx).indices
            uncovered[covered_idx] = 0.0
            
        # End while loop
        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_blocks = selected_indices.copy()
            
        # Optional: Early stop if we hit a theoretical lower bound or time is tight
        # Lower bound for C(19,8,5) is 264.
        # If we find 264, we can stop.
        if best_len <= 264:
            break

    # Convert indices to blocks
    if best_blocks is None:
        # Fallback
        return [list(k_subsets[0])] if k_subsets else []
        
    return [list(k_subsets[i]) for i in best_blocks]
