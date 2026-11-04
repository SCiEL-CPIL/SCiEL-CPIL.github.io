import numpy as np
import itertools
import time

_PRECOMP = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP:
        return _PRECOMP[key]
        
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Generate all t-subsets and map to index
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Precompute for each block, the indices of the t-subsets it contains
    # BlockT[i] is an array of t-subset indices for block i
    comb_k_t = int(np.math.comb(k, t))
    BlockT = np.zeros((n_blocks, comb_k_t), dtype=np.int32)
    
    # This loop is the bottleneck for precomputation
    # 75582 blocks * 56 subsets = 4.2M iterations
    # Using a list comprehension or vectorized approach?
    # itertools is C-implemented, so the inner loop is fast.
    for i, block in enumerate(all_blocks):
        # Get all t-subsets of this block
        # It's faster to iterate combinations directly
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            BlockT[i, j] = t_sub_index[t_sub]
            
    _PRECOMP[key] = (all_blocks, BlockT, n_t_subs)
    return _PRECOMP[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    all_blocks, BlockT, n_t_subs = _get_precomputed(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts
    # We can adjust this based on time taken for first run
    num_restarts = 20
    
    for run in range(num_restarts):
        if time.time() > deadline:
            break
            
        covered = np.zeros(n_t_subs, dtype=bool)
        blocks_idx = []
        
        while not covered.all():
            # Calculate gains
            # ~covered[BlockT] creates a boolean array of shape (n_blocks, comb(k,t))
            gains = (~covered[BlockT]).sum(axis=1)
            
            # Find max gain
            max_gain = gains.max()
            if max_gain == 0:
                break # Safety, shouldn't happen
            
            # Random tie-breaking among max gain blocks
            # This helps escape local optima
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[np.random.randint(len(candidates))]
            
            blocks_idx.append(idx)
            covered[BlockT[idx]] = True
            
        # Prune redundant blocks
        if blocks_idx:
            # Count coverage
            counts = np.zeros(n_t_subs, dtype=np.int32)
            for idx in blocks_idx:
                counts[BlockT[idx]] += 1
            
            pruned_idx = []
            # Remove in reverse order of addition
            for idx in reversed(blocks_idx):
                t_indices = BlockT[idx]
                counts[t_indices] -= 1
                # Check if any t-subset becomes uncovered
                if not (counts[t_indices] == 0).any():
                    pruned_idx.append(idx)
                else:
                    # Restore counts
                    counts[t_indices] += 1
                    
            if len(pruned_idx) < best_len:
                best_len = len(pruned_idx)
                best_blocks = pruned_idx
                
    if best_blocks is None:
        # Fallback: run one deterministic greedy
        covered = np.zeros(n_t_subs, dtype=bool)
        blocks_idx = []
        while not covered.all():
            gains = (~covered[BlockT]).sum(axis=1)
            idx = np.argmax(gains)
            blocks_idx.append(idx)
            covered[BlockT[idx]] = True
        
        # Prune
        counts = np.zeros(n_t_subs, dtype=np.int32)
        for idx in blocks_idx:
            counts[BlockT[idx]] += 1
        best_blocks = []
        for idx in reversed(blocks_idx):
            t_indices = BlockT[idx]
            counts[t_indices] -= 1
            if not (counts[t_indices] == 0).any():
                best_blocks.append(idx)
            else:
                counts[t_indices] += 1

    # Convert indices to lists of integers
    result = [list(all_blocks[i]) for i in best_blocks]
    return result
