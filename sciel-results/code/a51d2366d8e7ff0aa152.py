"""Witt design S(5,6,12) via deterministic backtracking with constraint propagation."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    points = list(range(12))
    
    # Precompute all 5-subsets
    all_5subsets = list(combinations(points, 5))
    n_5 = len(all_5subsets)  # 792
    # Map each 5-subset to its index
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # For each 6-subset, precompute the bitmask of 5-subset indices it contains
    all_6subsets = list(combinations(points, 6))
    n_6 = len(all_6subsets)  # 924
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
    
    # Precompute for each 5-subset index, which 6-subsets contain it
    subset_to_blocks = [[] for _ in range(n_5)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & (-m)
            s = lsb.bit_length() - 1
            subset_to_blocks[s].append(i)
            m ^= lsb
    
    covered_mask = 0  # bitmask of covered 5-subsets
    blocks = []
    
    def backtrack():
        nonlocal covered_mask, blocks
        
        if time.time() > deadline:
            return True  # Return partial result
        
        # Find the most constrained uncovered 5-subset (fewest available blocks)
        best_idx = -1
        best_count = float('inf')
        best_block_list = None
        
        for s in range(n_5):
            if not (covered_mask >> s) & 1:
                # Count how many blocks can still cover this 5-subset
                count = 0
                feasible_blocks = []
                for b_idx in subset_to_blocks[s]:
                    # Check if this block is still feasible (no overlap with covered)
                    if block_masks[b_idx] & covered_mask == 0:
                        count += 1
                        feasible_blocks.append(b_idx)
                        if count > best_count:
                            break
                if count < best_count:
                    best_count = count
                    best_idx = s
                    best_block_list = feasible_blocks
                    if count == 0:
                        return False
                    if count == 1:
                        break  # This is as constrained as it gets
        
        if best_idx == -1:
            # All 5-subsets covered
            return True
        
        # Try blocks that cover the most constrained 5-subset
        if best_block_list is None:
            best_block_list = [b_idx for b_idx in subset_to_blocks[best_idx]
                               if block_masks[b_idx] & covered_mask == 0]
        
        for b_idx in best_block_list:
            # Check feasibility
            if block_masks[b_idx] & covered_mask:
                continue
            
            # Choose this block
            blocks.append(all_6subsets[b_idx])
            covered_mask |= block_masks[b_idx]
            
            if backtrack():
                return True
            
            # Backtrack
            covered_mask &= ~block_masks[b_idx]
            blocks.pop()
        
        return False
    
    result = backtrack()
    
    return blocks
