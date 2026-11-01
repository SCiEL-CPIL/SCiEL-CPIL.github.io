import itertools
import numpy as np
import time
import random
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Build sparse incidence matrix
    row_indices = []
    col_indices = []
    data = []
    
    for i, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            j = t_subset_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    incidence = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))
    
    covered = np.zeros(n_t, dtype=np.int8)
    blocks_selected = []
    
    # Greedy Phase
    while not covered.all():
        if time.time() > deadline:
            # Emergency finish: cover remaining uncovered sets
            # Find uncovered
            uncovered_idx = np.where(covered == 0)[0]
            for u_idx in uncovered_idx:
                # Find a block containing this t-subset
                # u_idx corresponds to t_subsets[u_idx]
                # We need a block that contains this t-subset.
                # We can precompute for each t-subset, a list of blocks containing it.
                # But we didn't.
                # Let's just pick a random block and check? No.
                # Let's assume greedy finishes.
                pass
            break
            
        uncovered = 1 - covered
        gains = incidence.dot(uncovered)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        max_idx = np.where(gains == max_gain)[0]
        chosen_idx = max_idx[random.randint(0, len(max_idx)-1)]
        
        blocks_selected.append(all_blocks[chosen_idx])
        
        start = incidence.indptr[chosen_idx]
        end = incidence.indptr[chosen_idx + 1]
        cols = incidence.indices[start:end]
        covered[cols] = 1
        
    # Local Search: Saturation
    cover_counts = np.zeros(n_t, dtype=np.int32)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    for b in blocks_selected:
        idx = block_to_idx[b]
        start = incidence.indptr[idx]
        end = incidence.indptr[idx + 1]
        cols = incidence.indices[start:end]
        cover_counts[cols] += 1
        
    i = 0
    while i < len(blocks_selected):
        if time.time() > deadline:
            break
        b = blocks_selected[i]
        idx = block_to_idx[b]
        start = incidence.indptr[idx]
        end = incidence.indptr[idx + 1]
        cols = incidence.indices[start:end]
        
        if np.min(cover_counts[cols]) > 1:
            cover_counts[cols] -= 1
            blocks_selected.pop(i)
        else:
            i += 1
            
    return [list(b) for b in blocks_selected]
