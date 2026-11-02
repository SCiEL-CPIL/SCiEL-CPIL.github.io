import itertools
import time
import random
from collections import defaultdict, Counter

def _greedy_cover(v, k, t, rng, deadline):
    """
    Constructs a covering using a greedy heuristic with randomization.
    Uses a priority-based approach for selecting uncovered t-subsets.
    """
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset to start the block
        start_subset = rng.choice(tuple(uncovered))
        block = list(start_subset)
        block_set = set(block)
        
        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                break
                
            # Calculate gain for each candidate point
            best_gain = -1
            best_points = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check all (t-1)-subsets in block
                for sub in itertools.combinations(block, t - 1):
                    t_sub = tuple(sorted(sub + (x,)))
                    if t_sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            
            # If no gain, pick random point not in block
            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                x = rng.choice(best_points)
            
            block.append(x)
            block_set.add(x)
        
        block.sort()
        block_t = tuple(block)
        if block_t not in blocks:
            blocks.append(block_t)
        
        # Remove covered t-subsets
        for t_sub in itertools.combinations(block, t):
            uncovered.discard(tuple(t_sub))
            
    return blocks

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    Returns the reduced list of blocks.
    """
    if not blocks:
        return blocks
    
    # Map each t-subset to the list of blocks that cover it
    t_cover_count = defaultdict(int)
    
    # Build the map
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            t_cover_count[tuple(t_sub)] += 1
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            # Check if this block is essential
            is_essential = False
            for t_sub in itertools.combinations(block, t):
                if t_cover_count[tuple(t_sub)] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
            else:
                # Remove this block
                changed = True
                # Update cover counts
                for t_sub in itertools.combinations(block, t):
                    t_cover_count[tuple(t_sub)] -= 1
                # Do not add to new_blocks
        
        blocks = new_blocks
    
    return blocks

def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search: try to replace blocks with better ones.
    """
    if not blocks:
        return blocks
    
    # Build coverage map
    t_cover_count = defaultdict(int)
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            t_cover_count[tuple(t_sub)] += 1
    
    best_blocks = blocks[:]
    best_size = len(blocks)
    
    # Try removing non-essential blocks and see if we can re-cover with fewer
    # For now, just do the reduction pass again with different ordering
    for _ in range(10):
        if time.time() > deadline:
            break
        blocks = _reduce_covering(blocks, v, k, t)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
    
    return best_blocks

def _swap_blocks(blocks, v, k, t, rng, deadline):
    """
    Try to swap a block with a better one that covers more uncovered t-subsets.
    """
    if not blocks:
        return blocks
    
    # Build coverage map
    t_cover_count = defaultdict(int)
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            t_cover_count[tuple(t_sub)] += 1
    
    # Find t-subsets covered only once
    unique_t_subs = [t_sub for t_sub, count in t_cover_count.items() if count == 1]
    
    if not unique_t_subs:
        return blocks
    
    best_blocks = blocks[:]
    best_size = len(blocks)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Pick a random block
        idx = rng.randrange(len(blocks))
        old_block = blocks[idx]
        
        # Find t-subsets only covered by this block
        unique_to_block = [t_sub for t_sub in itertools.combinations(old_block, t) 
                          if t_cover_count[tuple(t_sub)] == 1]
        
        if not unique_to_block:
            continue
        
        # Try to find a new block that covers all unique_to_block t-subsets
        # and has minimal overlap with other blocks
        best_new_block = None
        best_score = float('inf')
        
        # Generate candidate blocks
        candidates = []
        for t_sub in unique_to_block:
            # Try to extend this t-subset to a k-block
            remaining = [x for x in range(v) if x not in t_sub]
            rng.shuffle(remaining)
            for r in range(min(5, len(remaining))):
                new_block = list(t_sub) + remaining[:k - t]
                new_block.sort()
                candidates.append(tuple(new_block))
        
        for cand in candidates:
            # Check if cand covers all unique_to_block
            cand_set = set(cand)
            covers_all = all(all(x in cand_set for x in t_sub) for t_sub in unique_to_block)
            if not covers_all:
                continue
            
            # Calculate score: number of t-subsets covered by cand that are already covered
            overlap = 0
            for t_sub in itertools.combinations(cand, t):
                if t_cover_count[tuple(t_sub)] > 0:
                    overlap += 1
            
            if overlap < best_score:
                best_score = overlap
                best_new_block = cand
        
        if best_new_block is not None:
            # Perform the swap
            blocks[idx] = best_new_block
            # Update coverage counts
            for t_sub in itertools.combinations(old_block, t):
                t_cover_count[tuple(t_sub)] -= 1
            for t_sub in itertools.combinations(best_new_block, t):
                t_cover_count[tuple(t_sub)] += 1
            
            # Try to reduce
            blocks = _reduce_covering(blocks, v, k, t)
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks[:]
    
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run to have a baseline
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    blocks = _local_search(blocks, v, k, t, rng, deadline)
    blocks = _swap_blocks(blocks, v, k, t, rng, deadline)
    best_blocks = blocks
    best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover(v, k, t, rng, deadline)
        
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        blocks = _local_search(blocks, v, k, t, rng, deadline)
        blocks = _swap_blocks(blocks, v, k, t, rng, deadline)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            if best_size <= 69:
                break
                
    return [list(b) for b in best_blocks]
