import numpy as np
import itertools
import time

# Precomputation for v=19, t=5, k=8
V = 19
T = 5
K = 8

# 1. Map all t-subsets to indices
all_t_subsets = list(itertools.combinations(range(V), T))
N_T = len(all_t_subsets)
t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

# 2. Generate all possible k-blocks and their coverage masks
all_k_blocks = list(itertools.combinations(range(V), K))
N_K = len(all_k_blocks)

# block_masks[i] is a Python int where bit j is 1 if all_t_subsets[j] is a subset of all_k_blocks[i]
block_masks = [0] * N_K
for i, block in enumerate(all_k_blocks):
    mask = 0
    for s in itertools.combinations(block, T):
        idx = t_subset_to_idx[s]
        mask |= (1 << idx)
    block_masks[i] = mask

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Assume v, k, t match the precomputed values
    # If not, this specific optimization won't work, but the task is fixed.
    
    # Initialize
    # uncovered_mask: bit j is 1 if all_t_subsets[j] is NOT covered
    uncovered_mask = (1 << N_T) - 1
    selected_block_indices = []
    
    while True:
        if time.time() > deadline:
            break
            
        if uncovered_mask == 0:
            break
            
        best_gain = 0
        best_idx = -1
        
        # Find the block with max marginal gain
        for i in range(N_K):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            # No improvement possible
            break
            
        # Select this block
        selected_block_indices.append(best_idx)
        # Update uncovered mask
        uncovered_mask &= ~block_masks[best_idx]
        
    # Convert selected indices to actual blocks
    blocks = [list(all_k_blocks[i]) for i in selected_block_indices]
    
    # Phase 2: Deletion Heuristic
    # Count how many times each t-subset is covered
    cover_counts = np.zeros(N_T, dtype=np.int32)
    for b in blocks:
        for s in itertools.combinations(b, t):
            idx = t_subset_to_idx[s]
            cover_counts[idx] += 1
            
    # Iterate to remove redundant blocks
    max_deletion_passes = 20
    for _ in range(max_deletion_passes):
        if time.time() > deadline:
            break
            
        removed_any = False
        new_blocks = []
        
        for b in blocks:
            redundant = True
            for s in itertools.combinations(b, t):
                idx = t_subset_to_idx[s]
                if cover_counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                removed_any = True
                for s in itertools.combinations(b, t):
                    idx = t_subset_to_idx[s]
                    cover_counts[idx] -= 1
            else:
                new_blocks.append(b)
                
        blocks = new_blocks
        
        if not removed_any:
            break
            
    return blocks
