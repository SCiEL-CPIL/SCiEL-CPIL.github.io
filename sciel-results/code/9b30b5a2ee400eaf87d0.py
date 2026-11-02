import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    def get_idx(s):
        return subset_to_idx[s]

    # Precompute all k-blocks and their covered subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_subs = []
    for b in all_blocks:
        subs = [get_idx(tuple(sorted(s))) for s in itertools.combinations(b, t)]
        block_subs.append(subs)
        
    # Inverse index: subset_idx -> list of block_idxs
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, b in enumerate(all_blocks):
        for s_idx in block_subs[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    # Greedy Initialization
    coverage_counts = [0] * num_subsets
    uncovered = set(range(num_subsets))
    blocks = []
    block_to_subsets = []
    
    while uncovered and time.time() < deadline:
        target_subset = next(iter(uncovered))
        candidates = subset_to_blocks[target_subset]
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidates:
            gain = 0
            for s_idx in block_subs[b_idx]:
                if s_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break
            
        blocks.append(all_blocks[best_block_idx])
        block_to_subsets.append(block_subs[best_block_idx])
        
        for s_idx in block_subs[best_block_idx]:
            coverage_counts[s_idx] += 1
            if coverage_counts[s_idx] == 1:
                uncovered.discard(s_idx)

    # Refinement
    def is_redundant(b_idx_list, counts):
        for s_idx in b_idx_list:
            if counts[s_idx] == 1:
                return False
        return True

    def remove_block(idx):
        for s_idx in block_to_subsets[idx]:
            coverage_counts[s_idx] -= 1
            if coverage_counts[s_idx] == 0:
                uncovered.add(s_idx)
        blocks.pop(idx)
        block_to_subsets.pop(idx)

    def add_block(b_tuple, b_subs):
        blocks.append(b_tuple)
        block_to_subsets.append(b_subs)
        for s_idx in b_subs:
            coverage_counts[s_idx] += 1
            if coverage_counts[s_idx] == 1:
                uncovered.discard(s_idx)

    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Remove redundant blocks
        for i in range(len(blocks)):
            if is_redundant(block_to_subsets[i], coverage_counts):
                remove_block(i)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Swap
        # Find block with fewest unique subsets
        best_swap_i = -1
        min_unique = float('inf')
        
        for i in range(len(blocks)):
            unique_count = 0
            for s_idx in block_to_subsets[i]:
                if coverage_counts[s_idx] == 1:
                    unique_count += 1
            if unique_count < min_unique:
                min_unique = unique_count
                best_swap_i = i
                
        if best_swap_i == -1 or min_unique == 0:
            if min_unique == 0:
                remove_block(best_swap_i)
                continue
            break # No blocks or stuck

        unique_subsets = [s_idx for s_idx in block_to_subsets[best_swap_i] if coverage_counts[s_idx] == 1]
        
        if not unique_subsets:
            continue

        # Find candidates covering all unique_subsets
        candidate_indices = set(subset_to_blocks[unique_subsets[0]])
        for s_idx in unique_subsets[1:]:
            candidate_indices.intersection_update(subset_to_blocks[s_idx])
            
        if not candidate_indices:
            # Try next best block? Or break.
            # Let's try to find another block to swap.
            # To avoid infinite loop, we can just break and return.
            # Or we can randomize.
            break

        active_blocks_set = set(blocks)
        best_new_block_idx = -1
        best_score = -1
        
        for b_idx in candidate_indices:
            if all_blocks[b_idx] in active_blocks_set:
                continue
            
            # Score: number of count-1 subsets covered
            score = 0
            for s_idx in block_subs[b_idx]:
                if coverage_counts[s_idx] == 1:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_new_block_idx = b_idx
                
        if best_new_block_idx != -1:
            remove_block(best_swap_i)
            add_block(all_blocks[best_new_block_idx], block_subs[best_new_block_idx])
            improved = True
            continue
            
        break # Stuck

    return [list(b) for b in blocks]
