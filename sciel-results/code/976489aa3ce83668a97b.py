import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # Precompute cover_matrix: for each block, the indices of t-subsets it covers
    block_cover_indices = []
    for block in all_k_subsets:
        t_subsets_in_block = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in t_subsets_in_block]
        block_cover_indices.append(indices)
        
    c_k_t = len(block_cover_indices[0])
    cover_matrix = np.array(block_cover_indices, dtype=np.int32)
    
    best_blocks = []
    best_count = float('inf')
    
    start_time = time.time()
    timeout = 115
    
    for restart in range(200):
        if time.time() - start_time > timeout:
            break
            
        random.seed(restart)
        
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_found_indices = []
        
        while np.any(uncovered):
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Allow slightly suboptimal choices for diversity
            candidates = np.where((gains >= max_gain - 1) & (gains > 0))[0]
            
            chosen_idx = candidates[random.randint(0, len(candidates)-1)]
            blocks_found_indices.append(chosen_idx)
            
            uncovered[cover_matrix[chosen_idx]] = False
            
        # Pruning
        if blocks_found_indices:
            counts = np.zeros(n_t_subsets, dtype=np.int32)
            for idx in blocks_found_indices:
                counts[cover_matrix[idx]] += 1
                
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks_found_indices) - 1, -1, -1):
                    block_idx = blocks_found_indices[i]
                    t_indices = cover_matrix[block_idx]
                    if np.min(counts[t_indices]) > 1:
                        counts[t_indices] -= 1
                        blocks_found_indices.pop(i)
                        changed = True
                        break
                        
            if len(blocks_found_indices) < best_count:
                best_count = len(blocks_found_indices)
                best_blocks = [all_k_subsets[i] for i in blocks_found_indices]

    return best_blocks
