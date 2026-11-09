import itertools
import numpy as np
import scipy.sparse
import time

# Precomputation
def precompute(v, k, t):
    # Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    rows = []
    cols = []
    data = []
    
    for i, block in enumerate(k_subsets):
        # Get all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            j = t_subsets_set[t_sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_k, n_t))
    return M, k_subsets, n_t

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute
    M, k_subsets, n_t = precompute(v, k, t)
    n_k = len(k_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Multi-start greedy with removal
    rng = np.random.default_rng(42)
    
    def greedy_with_removal(seed):
        local_rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t, dtype=np.int8)
        blocks = []
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() > deadline:
                return None
            
            # Calculate gains
            gains = M.dot(uncovered)
            
            if gains.max() == 0:
                break
                
            # Select top k
            top_k = 5
            top_k = min(top_k, n_k)
            indices = np.argpartition(gains, -top_k)[-top_k:]
            
            # Randomly pick one from top k
            best_idx = local_rng.choice(indices)
            
            blocks.append(best_idx)
            
            # Update cover_count and uncovered
            row = M[best_idx]
            idxs = row.indices
            cover_count[idxs] += 1
            uncovered[idxs] = 0
            
        # Removal Phase
        # Iterate until no more removals
        changed = True
        while changed:
            if time.time() > deadline:
                return None
            changed = False
            i = 0
            while i < len(blocks):
                b_idx = blocks[i]
                row = M[b_idx]
                idxs = row.indices
                
                # Check if redundant: no t-subset has count == 1
                # If any count is 1, it's essential
                if not np.any(cover_count[idxs] == 1):
                    # Remove it
                    cover_count[idxs] -= 1
                    blocks.pop(i)
                    changed = True
                    # Do not increment i
                else:
                    i += 1
                    
        if len(blocks) < best_len:
            return [k_subsets[i] for i in blocks]
        return None

    # Run multiple starts
    seed = 42
    while time.time() < deadline:
        res = greedy_with_removal(seed)
        if res is not None:
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
        seed += 1
        
    return best_blocks
