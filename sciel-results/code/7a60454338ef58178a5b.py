import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 5.0
    
    points = list(range(12))
    
    # Generate all 5-subsets and map to indices
    all_5_subsets = list(combinations(points, 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets) # 792
    
    # Generate all 6-subsets (potential blocks)
    all_6_subsets = list(combinations(points, 6))
    
    # Precompute bitmask for each 6-subset
    # Bit i is set if the 6-subset contains the 5-subset at index i
    block_masks = []
    for b in all_6_subsets:
        mask = 0
        # A 6-subset contains 6 5-subsets (remove one element each time)
        for i in range(6):
            s5 = tuple(sorted(b[:i] + b[i+1:]))
            mask |= 1 << subset_to_idx[s5]
        block_masks.append(mask)
        
    # Backtracking search for Exact Cover
    # We need to select 132 blocks such that their masks are disjoint and cover all 792 bits.
    # Total bits to cover: (1 << 792) - 1. This is too big for a single int in some contexts?
    # Python ints are arbitrary precision, so this is fine.
    
    target_mask = (1 << num_5_subsets) - 1
    best_blocks = []
    
    def backtrack(covered, current_blocks, start_idx):
        if time.time() > deadline:
            return False
        
        if covered == target_mask:
            best_blocks[:] = current_blocks
            return True
            
        # Heuristic: Find the 5-subset (bit) that is not covered yet,
        # and has the fewest candidate blocks covering it.
        # Instead of scanning all 792 bits, we can just pick the first uncovered bit?
        # That might be slow if the first bit has many options.
        # Better: Precompute for each 5-subset index, the list of block indices covering it.
        
        # Let's use a simpler heuristic first: Pick the first uncovered 5-subset.
        # Find first 0 bit in (target_mask ^ covered)? No, find first 1 bit in (target_mask & ~covered).
        uncovered_mask = target_mask & ~covered
        if uncovered_mask == 0:
            return len(current_blocks) == 132 # Should be exactly 132 if valid
            
        # Get index of least significant set bit in uncovered_mask
        # This corresponds to a specific 5-subset
        # (uncovered_mask & -uncovered_mask).bit_length() - 1
        idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        # Find all blocks that cover this 5-subset (idx) and are compatible with covered
        # We can iterate through all blocks, or precompute a list of blocks for each 5-subset.
        
        candidates = []
        for i in range(len(all_6_subsets)):
            if block_masks[i] & (1 << idx):
                if block_masks[i] & covered == 0:
                    candidates.append(i)
                    
        if not candidates:
            return False
            
        # Try each candidate
        for i in candidates:
            new_covered = covered | block_masks[i]
            current_blocks.append(all_6_subsets[i])
            if backtrack(new_covered, current_blocks, i+1):
                return True
            current_blocks.pop()
            
        return False

    # To make it faster, we can precompute `blocks_for_5subset[idx]`
    blocks_for_5subset = [[] for _ in range(num_5_subsets)]
    for i, mask in enumerate(block_masks):
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocks_for_5subset[idx].append(i)
            m ^= lsb

    # Refined Backtrack
    def backtrack_fast(covered, current_blocks):
        if time.time() > deadline:
            return False
        
        if covered == target_mask:
            best_blocks[:] = current_blocks
            return True
            
        uncovered_mask = target_mask & ~covered
        if uncovered_mask == 0:
            return len(current_blocks) == 132
            
        # Pick the 5-subset with fewest candidates
        # Heuristic: Check a few uncovered bits and pick the one with min candidates
        # Or just pick the first one for simplicity and rely on speed.
        # Let's pick the first one.
        idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        candidates = blocks_for_5subset[idx]
        
        # Filter compatible
        valid_cands = [i for i in candidates if block_masks[i] & covered == 0]
        
        if not valid_cands:
            return False
            
        for i in valid_cands:
            new_covered = covered | block_masks[i]
            current_blocks.append(all_6_subsets[i])
            if backtrack_fast(new_covered, current_blocks):
                return True
            current_blocks.pop()
            
        return False

    backtrack_fast(0, [])
    
    if len(best_blocks) == 132:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: Return empty or partial? 
        # If it fails, return whatever we have? 
        # The task says "Maximize". If we don't get 132, we get lower score.
        # But with bitmasks, it should find 132 instantly.
        return [list(b) for b in best_blocks]
