"""Improved covering algorithm with randomized restarts and redundancy removal."""
import itertools
import time
import random

def _greedy_cover(v, k, t, random_seed=None):
    """
    Constructs a covering using a greedy algorithm.
    If random_seed is None, uses deterministic selection (lexicographically smallest uncovered set).
    If random_seed is an int, uses random selection for the starting set of each block and tie-breaking.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = None

    while uncovered:
        # Select a seed t-set to start the new block
        if rng:
            # Convert to list for random choice. 
            # Optimization: For large v, this is costly, but for v=24 it's fine.
            seed = rng.choice(list(uncovered))
        else:
            seed = min(uncovered)
            
        block = list(seed)
        
        # Greedily add points to the block
        while len(block) < k:
            best_gain = -1
            best_points = []
            
            # Calculate gain for each candidate point not in block
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Only possible if we have enough elements to form t-1 subsets with x
                if len(block) >= t - 1:
                    for rest in itertools.combinations(block, t - 1):
                        # Create the candidate t-set
                        # rest is sorted, x is int. 
                        # We need the sorted tuple to match the keys in 'uncovered'
                        cand = tuple(sorted((x,) + rest))
                        if cand in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            
            # Select the next point
            if not best_points:
                # Fallback: pick smallest available point (should not happen if uncovered is not empty)
                available = [x for x in range(v) if x not in block]
                next_point = min(available) if available else 0
            elif rng and best_points:
                next_point = rng.choice(best_points)
            else:
                next_point = best_points[0]
            
            block.append(next_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for t_set in itertools.combinations(block, t):
            uncovered.discard(t_set)
            
    return blocks

def _remove_redundant(blocks, v, t):
    """
    Removes blocks that are redundant, i.e., every t-subset within them 
    is covered by other blocks.
    """
    if not blocks:
        return []
    
    # Count how many times each t-set is covered
    counts = {}
    for b in blocks:
        for t_set in itertools.combinations(b, t):
            counts[t_set] = counts.get(t_set, 0) + 1
    
    new_blocks = []
    # Iterate through blocks and try to remove them
    # Note: We must update counts as we remove blocks to ensure correctness for subsequent checks
    for b in blocks:
        removable = True
        for t_set in itertools.combinations(b, t):
            # If any t-set in this block is only covered by this block, we cannot remove it
            if counts.get(t_set, 0) <= 1:
                removable = False
                break
        
        if removable:
            # Remove the block and decrement counts for its t-sets
            for t_set in itertools.combinations(b, t):
                counts[t_set] -= 1
        else:
            new_blocks.append(b)
            
    return new_blocks

def construct(v, k, t):
    """
    Main entry point.
    Uses randomized restarts of the greedy algorithm followed by redundancy removal
    to find a minimal covering within the time limit.
    """
    start_time = time.time()
    # Leave a safety margin (10s) before the 150s limit
    deadline = start_time + 140
    
    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic run
    blocks = _greedy_cover(v, k, t, random_seed=None)
    blocks = _remove_redundant(blocks, v, t)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = blocks
    
    # 2. Randomized restarts
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        blocks = _greedy_cover(v, k, t, random_seed=seed_counter)
        blocks = _remove_redundant(blocks, v, t)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
