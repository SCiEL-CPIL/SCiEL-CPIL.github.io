import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    
    # 1. Enumerate all 5-subsets and assign indices
    all_5_subsets = list(combinations(points, 5))
    n_5 = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Enumerate all 6-subsets (blocks)
    all_blocks = list(combinations(points, 6))
    
    # 3. For each block, compute its mask of 5-subsets
    #    Also, create a list of blocks for each 5-subset index
    block_masks = []
    blocks_covering = [[] for _ in range(n_5)]
    
    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            blocks_covering[idx].append(len(block_masks))
        block_masks.append(mask)
        
    # 4. Solve Exact Cover
    # State: uncovered_mask (bits set for subsets that still need covering)
    # Initial state: all bits set (all 792 subsets need covering)
    # Goal: uncovered_mask == 0
    
    # Symmetry breaking:
    # The first block is fixed to (0, 1, 2, 3, 4, 5).
    # Find its index in all_blocks
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_blocks.index(first_block)
    
    # Initialize
    initial_uncovered = (1 << n_5) - 1
    initial_uncovered &= ~block_masks[first_block_idx]
    
    selected_blocks = [first_block]
    
    # Heuristic: Pick the 5-subset with the fewest candidate blocks (MRV)
    # We need to count how many *valid* blocks cover each subset.
    # A block is valid if it doesn't conflict with already selected blocks.
    # In the recursion, we maintain `uncovered_mask`.
    # A block `b` is valid if `block_masks[b] & ~uncovered_mask == 0` (i.e., block_mask is subset of uncovered_mask).
    
    def solve(uncovered_mask, selected_count):
        if time.time() > deadline:
            return None
            
        if uncovered_mask == 0:
            return selected_blocks[:]
            
        # Find the 5-subset with minimum remaining values (MRV)
        # Iterate through bits in uncovered_mask
        # We want the bit with the fewest blocks that can cover it without conflict.
        
        min_count = float('inf')
        best_subset_idx = -1
        best_candidates = []
        
        # Iterate over set bits in uncovered_mask
        m = uncovered_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            
            # Count valid blocks for this idx
            # A block is valid if its mask is a subset of uncovered_mask
            count = 0
            candidates = []
            for b_idx in blocks_covering[idx]:
                if block_masks[b_idx] & ~uncovered_mask == 0:
                    count += 1
                    candidates.append(b_idx)
                    
            if count < min_count:
                min_count = count
                best_subset_idx = idx
                best_candidates = candidates
                
            if min_count == 0:
                return None # Dead end
                
        # Try each candidate for the best_subset_idx
        for b_idx in best_candidates:
            # Select this block
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            selected_blocks.append(all_blocks[b_idx])
            
            result = solve(new_uncovered, selected_count + 1)
            if result is not None:
                return result
                
            # Backtrack
            selected_blocks.pop()
            
        return None

    result = solve(initial_uncovered, 1)
    
    if result is None:
        # If solver fails (shouldn't happen for S(5,6,12)), return empty or partial?
        # The task asks to maximize. If we fail, we return whatever we have?
        # But the solver is deterministic and should find it.
        # If timeout, return current best? The function returns a list.
        # If result is None, it means no solution found within time or dead end.
        # Since S(5,6,12) exists, it should find it.
        return []

    return result
