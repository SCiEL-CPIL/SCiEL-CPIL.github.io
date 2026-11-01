import itertools
import time
import numpy as np
import scipy.sparse as sp
import random

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_t_indices = []
    for block in all_k_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(subset_to_idx[t_sub])
        block_t_indices.append(indices)
    
    rows = []
    cols = []
    for i, indices in enumerate(block_t_indices):
        rows.extend([i] * len(indices))
        cols.extend(indices)
        
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n_k_blocks, n_t_subsets))
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        # Greedy Phase
        uncovered = np.ones(n_t_subsets, dtype=np.int32)
        used_block_indices = set()
        current_block_indices = []
        
        while uncovered.any():
            gains = M.dot(uncovered).toarray().flatten()
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            
            valid_candidates = [c for c in candidates if c not in used_block_indices]
            
            if not valid_candidates:
                all_nonzero = np.where(gains > 0)[0]
                valid_candidates = [c for c in all_nonzero if c not in used_block_indices]
                
            if not valid_candidates:
                break
            
            block_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
            
            used_block_indices.add(block_idx)
            current_block_indices.append(block_idx)
            
            block_sub_indices = block_t_indices[block_idx]
            uncovered[block_sub_indices] = 0
            
        if uncovered.any():
            continue
            
        # Removal Phase
        cover_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in current_block_indices:
            cover_counts[block_t_indices[b_idx]] += 1
        
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias?
            # random.shuffle(current_block_indices) 
            for b_idx in current_block_indices:
                removable = True
                for sub_idx in block_t_indices[b_idx]:
                    if cover_counts[sub_idx] <= 1:
                        removable = False
                        break
                
                if removable:
                    current_block_indices.remove(b_idx)
                    cover_counts[block_t_indices[b_idx]] -= 1
                    changed = True
        
        if len(current_block_indices) < best_len:
            # Verify
            final_cover = np.zeros(n_t_subsets, dtype=np.int8)
            for b_idx in current_block_indices:
                final_cover[block_t_indices[b_idx]] = 1
            if final_cover.all():
                best_len = len(current_block_indices)
                best_blocks = [all_k_blocks[i] for i in current_block_indices]
                
    if best_blocks is None:
        return []
        
    return best_blocks

# EVOLVE-BLOCK-END
