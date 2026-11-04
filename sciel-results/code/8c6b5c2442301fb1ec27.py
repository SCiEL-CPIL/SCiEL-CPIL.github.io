"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)
    
    # Track uncovered t-subsets
    uncovered = set(all_tsubsets)
    blocks = []
    
    # For efficiency, track how many times each point appears in uncovered t-subsets
    # and for pairs (if t >= 2), etc. But let's use a smarter greedy approach.
    
    # Key insight: when building a block, we want to maximize coverage of uncovered t-subsets.
    # A t-subset is covered by a block if it's contained in the block.
    # For a partial block B and candidate point x, the gain is the number of uncovered
    # t-subsets that contain x and (t-1) elements from B.
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset to seed a new block
        seed = min(uncovered)
        block = list(seed)
        
        # Build the rest of the block greedily
        while len(block) < k:
            if time.time() > deadline:
                break
            
            block_set = set(block)
            best_x = None
            best_gain = -1
            
            # Precompute the (t-1)-subsets of the current block
            if t >= 2:
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            else:
                t_minus_1_subsets = [()]
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in t_minus_1_subsets:
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove all t-subsets contained in this block from uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    # Convert to list of lists
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
