import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit

    points = range(12)
    blocks = list(combinations(points, 6))
    n_blocks = len(blocks)
    
    # Map 5-subsets to indices
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # Precompute masks for blocks
    # Each block covers 6 5-subsets
    block_masks = [0] * n_blocks
    for i, b in enumerate(blocks):
        mask = 0
        for s in combinations(b, 5):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Precompute for each 5-subset, which blocks cover it
    # bit_to_blocks[bit_idx] = list of block indices
    bit_to_blocks = [[] for _ in range(n_5subs)]
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        # Extract set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bit_to_blocks[idx].append(b_idx)
            m ^= lsb

    full_mask = (1 << n_5subs) - 1
    best_blocks = []
    best_count = 0
    found_optimal = False
    
    # Backtracking function
    # covered_mask: bitmask of covered 5-subsets
    # selected: list of selected block indices
    def backtrack(covered_mask, selected):
        nonlocal best_blocks, best_count, found_optimal
        
        if time.time() > deadline:
            return False
            
        # Check for optimal solution
        if len(selected) == 132:
            # Since no 5-subset is repeated, 132 blocks cover exactly 132*6 = 792 5-subsets.
            # There are only 792 5-subsets total.
            # So if we have 132 blocks, we must have covered all 792.
            # We can verify, but it's mathematically guaranteed if no overlap.
            if covered_mask == full_mask:
                best_blocks = selected[:]
                best_count = 132
                found_optimal = True
                return True
            else:
                # This shouldn't happen if logic is correct, but safe to check
                return False
                
        # Heuristic: MRV (Minimum Remaining Values)
        # Find the uncovered 5-subset with the fewest available blocks
        min_count = float('inf')
        min_bit = -1
        min_candidates = []
        
        # Iterate through all 5-subsets
        # Optimization: Only check uncovered bits
        remaining = full_mask ^ covered_mask
        # We can iterate through set bits of remaining
        # But finding the one with min candidates requires checking candidates for each.
        # Since n_5subs is 792, and each has ~7 candidates, this is fast.
        
        # To speed up, we can just iterate 0..791
        for bit_idx in range(n_5subs):
            if not (covered_mask & (1 << bit_idx)):
                # This bit is uncovered
                candidates = bit_to_blocks[bit_idx]
                # Count how many are compatible
                valid_candidates = []
                for b_idx in candidates:
                    if not (covered_mask & block_masks[b_idx]):
                        valid_candidates.append(b_idx)
                        
                count = len(valid_candidates)
                if count < min_count:
                    min_count = count
                    min_bit = bit_idx
                    min_candidates = valid_candidates
                    if count == 0:
                        break
                    if count == 1:
                        # Can't get better than 1 (unless 0, which we break on)
                        # But we might find 0 later, so we can't break immediately on 1
                        # However, if we find 0, we break.
                        pass
                        
        if min_count == 0:
            return False
            
        # Shuffle candidates to avoid bias
        random.shuffle(min_candidates)
        
        for b_idx in min_candidates:
            # Select b_idx
            selected.append(b_idx)
            new_covered = covered_mask | block_masks[b_idx]
            
            if backtrack(new_covered, selected):
                return True
                
            selected.pop()
            
        return False

    # Run the search
    # We can add a small amount of randomization to the initial state if it fails,
    # but MRV is usually deterministic enough to find the solution quickly if it exists.
    # If it times out, we return the best found so far.
    
    # To increase robustness, we can try a few different random shuffles of the block indices
    # if the first deterministic-like run (with shuffle inside) fails or is slow.
    # But the shuffle inside `backtrack` already provides randomization.
    
    backtrack(0, [])
    
    if found_optimal:
        return [list(blocks[i]) for i in best_blocks]
        
    # If not found, return best effort (might be empty or partial)
    # The problem asks to maximize. If we didn't find 132, we return what we have.
    # However, the backtrack function above only updates best_blocks if it finds 132.
    # We should update best_blocks during the search if we find a larger packing.
    
    # Let's modify the backtrack to track the best packing found so far.
    # Since we are doing DFS, we can update best_count whenever we select a block
    # that doesn't conflict, even if we don't reach 132.
    
    # Re-implementing with best tracking
    
    best_blocks = []
    best_count = 0
    found_optimal = False
    
    def backtrack2(covered_mask, selected):
        nonlocal best_blocks, best_count, found_optimal
        
        if time.time() > deadline:
            return False
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_blocks = selected[:]
            
        if found_optimal:
            return True
            
        if len(selected) == 132:
            if covered_mask == full_mask:
                best_blocks = selected[:]
                best_count = 132
                found_optimal = True
                return True
            return False
            
        # MRV Heuristic
        min_count = float('inf')
        min_bit = -1
        min_candidates = []
        
        for bit_idx in range(n_5subs):
            if not (covered_mask & (1 << bit_idx)):
                candidates = bit_to_blocks[bit_idx]
                valid_candidates = []
                for b_idx in candidates:
                    if not (covered_mask & block_masks[b_idx]):
                        valid_candidates.append(b_idx)
                        
                count = len(valid_candidates)
                if count < min_count:
                    min_count = count
                    min_bit = bit_idx
                    min_candidates = valid_candidates
                    if count == 0:
                        break
                        
        if min_count == 0:
            return False
            
        random.shuffle(min_candidates)
        
        for b_idx in min_candidates:
            selected.append(b_idx)
            new_covered = covered_mask | block_masks[b_idx]
            
            if backtrack2(new_covered, selected):
                return True
                
            selected.pop()
            
        return False

    backtrack2(0, [])
    
    return [list(blocks[i]) for i in best_blocks]
