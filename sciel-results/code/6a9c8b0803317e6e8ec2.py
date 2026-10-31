import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # 2. Weighted Greedy Set Cover
    # Weight of a t-subset is inversely proportional to the number of blocks covering it.
    # This helps cover "rare" subsets earlier.
    weights = [0.0] * num_t_subsets
    for t_idx in range(num_t_subsets):
        count = len(t_subset_blocks[t_idx])
        if count > 0:
            weights[t_idx] = 1.0 / count
        else:
            weights[t_idx] = 0.0 # Should not happen if v,k,t are valid
    
    uncovered = [True] * num_t_subsets
    gain = [0.0] * num_k_subsets
    
    # Initialize gain
    for b_idx in range(num_k_subsets):
        g = 0.0
        for t_idx in block_t_indices[b_idx]:
            g += weights[t_idx]
        gain[b_idx] = g
        
    selected_blocks = []
    
    while True:
        max_gain = max(gain)
        if max_gain == 0:
            break
            
        # Find block with max gain
        best_block = gain.index(max_gain)
        
        selected_blocks.append(best_block)
        
        # Update gains
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                w = weights[t_idx]
                # Decrement gain for all blocks covering this t_subset
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= w
                    
    # 3. Iterative Minimalization (Deletion)
    # Calculate cover counts
    cover_count = [0] * num_t_subsets
    for b_idx in selected_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    active_blocks = set(selected_blocks)
    
    # Iterative deletion pass
    # We repeat until no more blocks can be removed
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias in which block gets removed first
        indices_to_check = list(range(len(selected_blocks)))
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            b_idx = selected_blocks[i]
            if b_idx not in active_blocks:
                continue
                
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                changed = True
                
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    
    return result
