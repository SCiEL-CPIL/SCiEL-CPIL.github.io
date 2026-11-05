import itertools
import numpy as np
from math import comb
import scipy.sparse as sp

def construct(v, k, t):
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Build sparse matrix
    row_data = []
    col_data = []
    
    # Local vars for speed
    tti = triple_to_idx
    comb_func = itertools.combinations
    
    for i, block in enumerate(all_blocks):
        for sub in comb_func(block, t):
            row_data.append(i)
            col_data.append(tti[sub])
            
    row_arr = np.array(row_data, dtype=np.int32)
    col_arr = np.array(col_data, dtype=np.int32)
    data_arr = np.ones(len(row_arr), dtype=np.int8)
    
    block_matrix = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(n_blocks, n_triples)).tocsr()
    
    # Store block tuples for output
    # all_blocks is already a list of tuples
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_vec = np.ones(n_triples, dtype=np.int8)
        selected_blocks = []
        remaining = n_triples
        
        # Local refs
        bm = block_matrix
        ab = all_blocks
        
        while remaining > 0:
            gains = bm.dot(uncovered_vec)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update uncovered
            triple_indices = bm[best_idx].indices
            uncovered_vec[triple_indices] = 0
            remaining -= max_gain
            selected_blocks.append(list(ab[best_idx]))
            
        return selected_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Increase restarts
    n_restarts = 500
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
