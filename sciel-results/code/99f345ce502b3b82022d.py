import itertools
import time
import numpy as np
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    limit = 140.0 # Safety margin
    
    # 1. Generate all t-subsets and map to index
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = set(t_subsets)
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Precompute t-subset indices for each block
    block_t_indices = []
    block_masks = [0] * num_k
    
    # Initialize degrees
    current_degree = np.zeros(num_t, dtype=np.int32)
    
    # Data for sparse matrix
    row_data = []
    col_data = []
    
    for idx, block in enumerate(k_blocks):
        indices = []
        mask = 0
        for combo in itertools.combinations(block, t):
            t_idx = t_index_map[combo]
            indices.append(t_idx)
            mask |= (1 << t_idx)
            current_degree[t_idx] += 1
            row_data.append(idx)
            col_data.append(t_idx)
        block_t_indices.append(indices)
        block_masks[idx] = mask
        
    # Initialize weights
    weights = np.zeros(num_t, dtype=np.float64)
    for i in range(num_t):
        if current_degree[i] > 0:
            weights[i] = 1.0 / current_degree[i]
        else:
            weights[i] = 0.0
            
    # Create Sparse Matrix M
    # data is all 1s
    data = np.ones(len(row_data), dtype=np.float32)
    M = sp.coo_matrix((data, (row_data, col_data)), shape=(num_k, num_t)).tocsr()
    
    # Greedy Set Cover
    uncovered_mask = (1 << num_t) - 1
    selected_indices = []
    selected_set = set()
    
    while uncovered_mask:
        if time.time() - start_time > limit:
            break
            
        # Calculate gains: M @ weights
        gains = M.dot(weights)
        
        # Find best block
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        
        if best_gain <= 0:
            break
            
        # Select best_idx
        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        
        # Update uncovered mask
        uncovered_mask &= ~block_masks[best_idx]
        
        # Update degrees and weights
        for t_idx in block_t_indices[best_idx]:
            current_degree[t_idx] -= 1
            if current_degree[t_idx] > 0:
                weights[t_idx] = 1.0 / current_degree[t_idx]
            else:
                weights[t_idx] = 0.0
                
    # 3. Post-processing: Remove Redundant Blocks
    if selected_indices:
        changed = True
        while changed and (time.time() - start_time < limit):
            changed = False
            new_selected = []
            for i in range(len(selected_indices)):
                idx_i = selected_indices[i]
                others_mask = 0
                for j in range(len(selected_indices)):
                    if i != j:
                        others_mask |= block_masks[selected_indices[j]]
                        
                if (block_masks[idx_i] & ~others_mask) == 0:
                    changed = True
                else:
                    new_selected.append(idx_i)
            if changed:
                selected_indices = new_selected
                
    return [k_blocks[i] for i in selected_indices]
