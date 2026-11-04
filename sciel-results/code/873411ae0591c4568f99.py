import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage as numpy arrays
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build coverage matrix: cover_matrix[i, j] = 1 if block i covers t-subset j
    # Use a sparse-like approach with numpy for vectorized gain computation
    # For n_t_subs ~ 3876 and n_k_blocks ~ 50388, this is manageable
    
    # First, build a list of t-subset indices for each block
    block_t_indices = []
    for block in all_k_blocks:
        t_indices = [sub_to_idx[t_sub] for t_sub in itertools.combinations(block, t)]
        block_t_indices.append(t_indices)
    
    # Build numpy coverage matrix (n_k_blocks x n_t_subs) with 0s and 1s
    cover_matrix = np.zeros((n_k_blocks, n_t_subs), dtype=np.uint8)
    for i, t_indices in enumerate(block_t_indices):
        cover_matrix[i, t_indices] = 1
    
    # 3. Greedy Construction with vectorized gain computation
    uncovered = np.ones(n_t_subs, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.any():
        if time.time() > deadline - 30:
            break
        
        # Vectorized gain computation: count how many uncovered subsets each block covers
        gains = (cover_matrix & uncovered[np.newaxis, :]).sum(axis=1)
        
        best_idx = int(np.argmax(gains))
        best_gain = int(gains[best_idx])
        
        if best_gain == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered[cover_matrix[best_idx] == 1] = 0
    
    # 4. Local Search: Redundancy Removal
    # Build cover counts
    cover_count = np.zeros(n_t_subs, dtype=np.int32)
    for idx in selected_indices:
        cover_count[cover_matrix[idx] == 1] += 1
    
    current_blocks = selected_indices[:]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_blocks = []
        
        for i in range(len(current_blocks)):
            idx = current_blocks[i]
            mask_positions = np.where(cover_matrix[idx] == 1)[0]
            
            # Check if this block is the only cover for any subset
            is_critical = np.any(cover_count[mask_positions] == 1)
            
            if is_critical:
                new_blocks.append(idx)
            else:
                # Remove this block
                cover_count[mask_positions] -= 1
                improved = True
        
        current_blocks = new_blocks
    
    # 5. Swap-based local search: try replacing one block with another to reduce count
    # This is more complex; for now, focus on the improved greedy + pruning
    
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
