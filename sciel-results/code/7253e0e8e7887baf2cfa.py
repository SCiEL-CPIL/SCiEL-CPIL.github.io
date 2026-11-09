"""Improved program for covering design C(19,7,4) using bitmask-based vectorized greedy."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin from 150s limit

    # Precompute all t-subsets as tuples
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Create a mapping from t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Initialize uncovered mask: 1 means uncovered
    uncovered_mask = np.ones(n_subsets, dtype=np.int32)
    
    # Precompute for each point x and each subset of points of size t-1, 
    # which t-subsets would be covered if we add x to that subset
    # Actually, let's think differently:
    # For a block B, the t-subsets it covers are all t-subsets contained in B.
    # We want to greedily build blocks to maximize coverage.
    
    # Better approach: For each possible block (7-subset), precompute which t-subsets it covers.
    # But there are C(19,7) = 50388 blocks, and each covers C(7,4) = 35 t-subsets.
    # We can precompute a matrix: for each block, a bitmask of which t-subsets it covers.
    
    # Actually, with n_subsets = C(19,4) = 3876, we can use numpy arrays.
    
    # Precompute: for each 7-subset (block), which 4-subsets does it contain?
    # There are C(19,7) = 50388 blocks. For each, compute a boolean array of length 3876.
    # That's 50388 * 3876 ≈ 195 million entries, which is too much memory.
    
    # Alternative: use a different strategy.
    # Greedy: at each step, find the block that covers the most currently uncovered t-subsets.
    # But checking all 50388 blocks each time is expensive.
    
    # Better greedy: build blocks incrementally.
    # Start with an uncovered t-subset, then add points to maximize new coverage.
    
    # Let's use the incremental approach but with numpy for speed.
    
    blocks = []
    
    # For efficiency, track which t-subsets are uncovered using a set
    uncovered_set = set(all_t_subsets)
    
    while uncovered_set:
        if time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset (deterministic)
        base_subset = min(uncovered_set)
        base_set = set(base_subset)
        
        # Start building a block from this subset
        current_block = list(base_subset)
        current_block_set = set(current_block)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(current_block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
            # For each candidate point not in current block, compute how many new t-subsets it covers
            for x in range(v):
                if x in current_block_set:
                    continue
                
                # New t-subsets covered by adding x: 
                # t-subsets that contain x and t-1 elements from current_block
                # and are currently uncovered
                gain = 0
                for combo in itertools.combinations(current_block, t - 1):
                    t_sub = tuple(sorted(combo + (x,)))
                    if t_sub in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                # No point adds coverage, break
                break
            
            current_block.append(best_point)
            current_block_set.add(best_point)
        
        # If we couldn't fill the block, fill with arbitrary points
        if len(current_block) < k:
            remaining = [x for x in range(v) if x not in current_block_set]
            for x in remaining[:k - len(current_block)]:
                current_block.append(x)
                current_block_set.add(x)
        
        current_block.sort()
        blocks.append(current_block)
        
        # Update uncovered set: remove all t-subsets contained in this block
        for t_sub in itertools.combinations(current_block, t):
            uncovered_set.discard(t_sub)
    
    # If we ran out of time, we need to ensure validity.
    # Check if all t-subsets are covered.
    if uncovered_set:
        # Emergency: cover remaining t-subsets with minimal blocks
        while uncovered_set:
            if time.time() > deadline - 2:
                # Last resort: add blocks for each uncovered t-subset
                for t_sub in list(uncovered_set):
                    if time.time() > deadline:
                        break
                    block = list(t_sub)
                    block_set = set(block)
                    for x in range(v):
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    block.sort()
                    blocks.append(block)
                    for ts in itertools.combinations(block, t):
                        uncovered_set.discard(ts)
                break
            
            # Normal greedy for remaining
            base_subset = min(uncovered_set)
            base_set = set(base_subset)
            current_block = list(base_subset)
            current_block_set = set(current_block)
            
            while len(current_block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in current_block_set:
                        continue
                    gain = 0
                    for combo in itertools.combinations(current_block, t - 1):
                        t_sub = tuple(sorted(combo + (x,)))
                        if t_sub in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1 or best_gain == 0:
                    break
                current_block.append(best_point)
                current_block_set.add(best_point)
            
            if len(current_block) < k:
                remaining = [x for x in range(v) if x not in current_block_set]
                for x in remaining[:k - len(current_block)]:
                    current_block.append(x)
                    current_block_set.add(x)
            
            current_block.sort()
            blocks.append(current_block)
            for t_sub in itertools.combinations(current_block, t):
                uncovered_set.discard(t_sub)
    
    return blocks
# EVOLVE-BLOCK-END
