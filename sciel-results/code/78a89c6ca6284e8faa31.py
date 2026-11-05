import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Use 145s to be safe against 150s limit
    deadline = start_time + 145.0
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    block_t_indices = [[] for _ in range(n_k_blocks)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices

    t_sub_to_blocks = [[] for _ in range(n_t_subs)]
    for i in range(n_k_blocks):
        for idx in block_t_indices[i]:
            t_sub_to_blocks[idx].append(i)

    def greedy_cover(randomize=False):
        uncovered_mask = (1 << n_t_subs) - 1
        selected_blocks = []
        
        while uncovered_mask:
            # Find lowest set bit index
            lowbit = uncovered_mask & -uncovered_mask
            target_idx = lowbit.bit_length() - 1
            
            # Get candidates covering target_idx
            # Copy list to allow shuffling
            candidates = t_sub_to_blocks[target_idx][:]
            
            if randomize:
                random.shuffle(candidates)
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = (uncovered_mask & block_masks[b_idx]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    # Early termination if we cover all subsets of this block?
                    # If gain == len(block_t_indices[b_idx]), it's max for this block.
                    # But another block might have higher gain.
                    # So we must check all candidates.
            
            selected_blocks.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return selected_blocks

    def remove_redundant(blocks):
        # Calculate counts
        counts = [0] * n_t_subs
        for b in blocks:
            for idx in block_t_indices[b]:
                counts[idx] += 1
        
        new_blocks = []
        changed = False
        for b in blocks:
            redundant = True
            for idx in block_t_indices[b]:
                if counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_t_indices[b]:
                    counts[idx] -= 1
                changed = True
            else:
                new_blocks.append(b)
        return new_blocks, changed

    def local_search(blocks):
        # Iterative improvement with swaps
        # Limit iterations to prevent hanging on a single instance
        max_iters = 100
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            blocks, changed = remove_redundant(blocks)
            if changed:
                continue
            
            if not blocks:
                break
                
            # Try a swap
            # Pick a random block
            i = random.randint(0, len(blocks)-1)
            b = blocks[i]
            
            # Find unique t-subsets for b
            counts = [0] * n_t_subs
            for bb in blocks:
                for idx in block_t_indices[bb]:
                    counts[idx] += 1
            
            unique_indices = [idx for idx in block_t_indices[b] if counts[idx] == 1]
            
            if not unique_indices:
                # Should have been removed, but just in case
                blocks.pop(i)
                continue
            
            # Find candidate replacements
            if len(unique_indices) == 1:
                cand_list = t_sub_to_blocks[unique_indices[0]]
            else:
                cand_set = set(t_sub_to_blocks[unique_indices[0]])
                for idx in unique_indices[1:]:
                    cand_set.intersection_update(t_sub_to_blocks[idx])
                cand_list = list(cand_set)
            
            blocks_set = set(blocks)
            valid_cands = [c for c in cand_list if c not in blocks_set]
            
            if valid_cands:
                new_b = random.choice(valid_cands)
                blocks[i] = new_b
                # We don't immediately remove redundant here to keep the loop simple
                # The next iteration of the for-loop will call remove_redundant
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run initial deterministic
    blocks = greedy_cover(randomize=False)
    blocks = local_search(blocks)
    best_len = len(blocks)
    best_blocks = blocks.copy()
    
    # Run randomized instances
    while time.time() < deadline:
        blocks = greedy_cover(randomize=True)
        blocks = local_search(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in best_blocks]
