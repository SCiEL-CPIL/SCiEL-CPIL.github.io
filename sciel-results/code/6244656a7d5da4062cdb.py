import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0
    
    points = range(12)
    
    # Generate all 5-subsets and map them to indices
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets) # 792
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Generate all 6-subsets
    all_6subsets = list(combinations(points, 6))
    num_blocks = len(all_6subsets) # 924
    
    # Pre-compute coverage masks for each block
    # block_cover[i] is an integer where bit j is 1 if 5-subset j is in block i
    block_cover = [0] * num_blocks
    block_list = [0] * num_blocks # Store the actual subset for output
    
    for i, b in enumerate(all_6subsets):
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
        block_cover[i] = mask
        block_list[i] = b
        
    # Symmetry breaking: Fix first block to be (0, 1, 2, 3, 4, 5)
    first_block_idx = None
    for i, b in enumerate(all_6subsets):
        if b == (0, 1, 2, 3, 4, 5):
            first_block_idx = i
            break
            
    if first_block_idx is None:
        first_block_idx = 0
        
    initial_mask = block_cover[first_block_idx]
    
    best_solution = []
    
    def backtrack(used_mask, current_blocks, count):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        if count == 132:
            best_solution = current_blocks[:]
            return True
            
        # Heuristic: Find a 5-subset that is not covered yet and has fewest candidate blocks
        # To do this efficiently, we can look at the bits in ~used_mask.
        # But 792 bits is large.
        # Alternative: Just iterate through all blocks and check if they are valid.
        # Since we are looking for 132 blocks, and the space is small, this might be fast enough.
        
        # Better Heuristic: 
        # Pick an uncovered 5-subset.
        # Find all blocks that cover it and are valid.
        # If none, fail.
        # Pick the one with minimum options.
        
        # To find an uncovered 5-subset, we can iterate through the 792 indices.
        # This is 792 iterations per node.
        # Depth is 132.
        # 792 * 132 * branching_factor.
        
        # Let's try a simpler heuristic first: 
        # Iterate blocks in index order.
        # This might be slow if the search tree is wide.
        
        # Let's use the "most constrained" variable heuristic.
        # We need to know, for each uncovered 5-subset, how many valid blocks cover it.
        # We can maintain a count array.
        
        # For now, let's just try to find ANY valid block.
        # If we find 132, we return.
        
        for i in range(num_blocks):
            if i == first_block_idx and count == 0:
                continue # Already applied? No, we pass used_mask.
                
            if (block_cover[i] & used_mask) == 0:
                # Valid block
                new_mask = used_mask | block_cover[i]
                if backtrack(new_mask, current_blocks + [i], count + 1):
                    return True
                    
        return False
        
    # Start backtracking
    # We already applied the first block in initial_mask?
    # No, let's start with count=1 and used_mask=initial_mask.
    if backtrack(initial_mask, [first_block_idx], 1):
        return [list(all_6subsets[i]) for i in best_solution]
    else:
        # Fallback: Greedy
        used_mask = initial_mask
        blocks = [first_block_idx]
        for i in range(num_blocks):
            if i == first_block_idx:
                continue
            if (block_cover[i] & used_mask) == 0:
                used_mask |= block_cover[i]
                blocks.append(i)
                if len(blocks) == 132:
                    break
        return [list(all_6subsets[i]) for i in blocks]
