import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    N_k = len(all_k_subsets)
    
    block_masks = [0] * N_k
    block_t_indices = [[] for _ in range(N_k)]
    
    for i, block in enumerate(all_k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices

    initial_uncovered_mask = (1 << N_t) - 1
    best_blocks = None
    best_count = float('inf')

    # 2. Main Loop
    while time.time() < deadline:
        uncovered_mask = initial_uncovered_mask
        selected_blocks = []
        cover_counts = [0] * N_t
        
        # Greedy Phase
        while uncovered_mask:
            max_gain = 0
            best_indices = []
            
            for i in range(N_k):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_indices = [i]
                elif gain == max_gain:
                    best_indices.append(i)
            
            if not best_indices:
                break
                
            chosen_idx = random.choice(best_indices)
            selected_blocks.append(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
            for idx in block_t_indices[chosen_idx]:
                cover_counts[idx] += 1

        # Redundancy Removal - single pass, no restart
        for i in range(len(selected_blocks) - 1, -1, -1):
            block_idx = selected_blocks[i]
            
            for t_idx in block_t_indices[block_idx]:
                cover_counts[t_idx] -= 1
            
            if min(cover_counts) > 0:
                selected_blocks.pop(i)
            else:
                for t_idx in block_t_indices[block_idx]:
                    cover_counts[t_idx] += 1

        # Second redundancy pass (catches cascading removals)
        for i in range(len(selected_blocks) - 1, -1, -1):
            block_idx = selected_blocks[i]
            
            for t_idx in block_t_indices[block_idx]:
                cover_counts[t_idx] -= 1
            
            if min(cover_counts) > 0:
                selected_blocks.pop(i)
            else:
                for t_idx in block_t_indices[block_idx]:
                    cover_counts[t_idx] += 1

        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = [list(all_k_subsets[i]) for i in selected_blocks]

    if best_blocks is None:
        uncovered_mask = initial_uncovered_mask
        selected_blocks = []
        while uncovered_mask:
            max_gain = 0
            best_idx = 0
            for i in range(N_k):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        best_blocks = [list(all_k_subsets[i]) for i in selected_blocks]

    return best_blocks
