import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute list of subset indices for each block
    block_subset_indices = []
    for b in all_blocks:
        subs = itertools.combinations(b, t)
        indices = [subset_to_idx[s] for s in subs]
        block_subset_indices.append(indices)
        
    # Precompute bitsets for each block
    block_bitsets = []
    for indices in block_subset_indices:
        bits = 0
        for idx in indices:
            bits |= (1 << idx)
        block_bitsets.append(bits)
        
    # Precompute subset to blocks mapping for fast repair
    subset_to_blocks = [[] for _ in range(total_subsets)]
    for bi, indices in enumerate(block_subset_indices):
        for idx in indices:
            subset_to_blocks[idx].append(bi)

    mask_all = (1 << total_subsets) - 1

    def greedy_construct(randomize=False):
        uncovered_bits = mask_all
        current_blocks = []
        used_blocks = [False] * num_blocks
        
        while uncovered_bits != 0:
            best_gain = 0
            best_candidates = []
            
            for bi in range(num_blocks):
                if used_blocks[bi]:
                    continue
                gain = (uncovered_bits & block_bitsets[bi]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [bi]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(bi)
            
            if not best_candidates:
                break
                
            if randomize and len(best_candidates) > 1:
                best_block_idx = random.choice(best_candidates)
            else:
                best_block_idx = best_candidates[0]
                
            current_blocks.append(best_block_idx)
            used_blocks[best_block_idx] = True
            uncovered_bits &= ~block_bitsets[best_block_idx]
            
        return current_blocks

    def calculate_counts(blocks):
        counts = np.zeros(total_subsets, dtype=np.int32)
        for bi in blocks:
            for idx in block_subset_indices[bi]:
                counts[idx] += 1
        return counts

    def remove_redundant(blocks, counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                bi = blocks[i]
                redundant = True
                for s_idx in block_subset_indices[bi]:
                    if counts[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for s_idx in block_subset_indices[bi]:
                        counts[s_idx] -= 1
                    changed = True
        return blocks

    def is_valid_swap(old_bi, new_bi, counts):
        for s_idx in block_subset_indices[old_bi]:
            if counts[s_idx] == 1:
                if (block_bitsets[new_bi] & (1 << s_idx)) == 0:
                    return False
        return True

    def local_improve(blocks):
        blocks = list(blocks)
        counts = calculate_counts(blocks)
        blocks = remove_redundant(blocks, counts)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                old_bi = blocks[i]
                for _ in range(20): # Reduced swaps for speed
                    new_bi = random.randint(0, num_blocks - 1)
                    if new_bi in blocks:
                        continue
                    if is_valid_swap(old_bi, new_bi, counts):
                        blocks[i] = new_bi
                        for s_idx in block_subset_indices[old_bi]:
                            counts[s_idx] -= 1
                        for s_idx in block_subset_indices[new_bi]:
                            counts[s_idx] += 1
                        
                        prev_len = len(blocks)
                        blocks = remove_redundant(blocks, counts)
                        if len(blocks) < prev_len:
                            improved = True
                            break
                if improved:
                    break
        return blocks

    def repair(blocks):
        blocks = list(blocks)
        counts = calculate_counts(blocks)
        uncovered_set = {i for i in range(total_subsets) if counts[i] == 0}
        
        while uncovered_set:
            target_idx = random.choice(list(uncovered_set))
            candidates = subset_to_blocks[target_idx]
            block_set = set(blocks)
            valid_candidates = [b for b in candidates if b not in block_set]
            
            if not valid_candidates:
                break
                
            new_bi = random.choice(valid_candidates)
            blocks.append(new_bi)
            
            for s_idx in block_subset_indices[new_bi]:
                counts[s_idx] += 1
                if counts[s_idx] == 1:
                    uncovered_set.remove(s_idx)
                    
        return blocks

    start_time = time.time()
    time_limit = 115
    
    best_blocks = greedy_construct(randomize=False)
    best_len = len(best_blocks)
    
    current_blocks = greedy_construct(randomize=True)
    
    while time.time() - start_time < time_limit:
        current_blocks = local_improve(current_blocks)
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        perturb_amount = max(1, int(0.1 * len(current_blocks)))
        perturbed = list(current_blocks)
        
        for _ in range(perturb_amount):
            if not perturbed:
                break
            idx_to_remove = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx_to_remove)
            
        current_blocks = repair(perturbed)

    return [list(all_blocks[bi]) for bi in best_blocks]
