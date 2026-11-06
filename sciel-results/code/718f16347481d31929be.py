import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = range(12)
    
    # 1. Enumerate all 5-subsets
    all_5_subsets = list(combinations(points, 5))
    n_5 = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Enumerate all 6-subsets
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # 3. Compute masks and reverse index
    block_masks = [0] * n_blocks
    blocks_covering = [[] for _ in range(n_5)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            blocks_covering[idx].append(i)
        block_masks[i] = mask
    
    # Symmetry breaking: Fix first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_blocks.index(first_block)
    
    initial_uncovered = (1 << n_5) - 1
    initial_uncovered &= ~block_masks[first_block_idx]
    
    selected_blocks = [first_block]
    
    # Greedy pre-selection to reduce search space
    greedy_selected = [first_block_idx]
    greedy_selected_set = {first_block_idx}
    greedy_uncovered = initial_uncovered
    
    # Precompute block coverage counts
    block_cover_count = [bin(block_masks[i]).count('1') for i in range(n_blocks)]
    
    # Greedy: repeatedly pick the block covering most uncovered 5-subsets
    while greedy_uncovered != 0:
        if time.time() > deadline - 10.0:
            break
        
        best_block = -1
        best_count = -1
        
        for i in range(n_blocks):
            if i in greedy_selected_set:
                continue
            # Count how many of this block's 5-subsets are still uncovered
            count = bin(block_masks[i] & greedy_uncovered).count('1')
            if count > best_count:
                best_count = count
                best_block = i
        
        if best_block == -1 or best_count == 0:
            break
            
        greedy_selected.append(best_block)
        greedy_selected_set.add(best_block)
        greedy_uncovered &= ~block_masks[best_block]
    
    # If greedy found all 132 blocks, return immediately
    if greedy_uncovered == 0 and len(greedy_selected) == 132:
        return [all_blocks[i] for i in greedy_selected]
    
    def solve(uncovered_mask, selected_indices, selected_set, selected_count):
        if time.time() > deadline:
            return None
            
        if uncovered_mask == 0:
            return [all_blocks[i] for i in selected_indices]
            
        # MRV: Find 5-subset with min candidates
        min_count = float('inf')
        best_subset_idx = -1
        best_candidates = []
        
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            
            count = 0
            candidates = []
            for b_idx in blocks_covering[idx]:
                if b_idx in selected_set:
                    continue
                if block_masks[b_idx] & ~uncovered_mask == 0:
                    count += 1
                    candidates.append(b_idx)
                    
            if count < min_count:
                min_count = count
                best_subset_idx = idx
                best_candidates = candidates
                
            if min_count == 0:
                return None
                
        # Backtrack
        for b_idx in best_candidates:
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            selected_indices.append(b_idx)
            selected_set.add(b_idx)
            
            result = solve(new_uncovered, selected_indices, selected_set, selected_count + 1)
            if result is not None:
                return result
                
            selected_indices.pop()
            selected_set.discard(b_idx)
            
        return None

    # Try backtracking from the greedy solution's state if it got close
    if len(greedy_selected) >= 125:
        result = solve(greedy_uncovered, greedy_selected[:], set(greedy_selected), len(greedy_selected))
        if result is not None:
            return result
    
    # Fall back to original backtracking from just the first block
    result = solve(initial_uncovered, [first_block_idx], {first_block_idx}, 1)
    
    if result is None:
        # Return the best we have
        if greedy_uncovered == 0:
            return [all_blocks[i] for i in greedy_selected]
        return [all_blocks[i] for i in greedy_selected]
        
    return result
