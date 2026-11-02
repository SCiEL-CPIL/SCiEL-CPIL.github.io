import itertools
import time
import random
from collections import defaultdict

def _greedy_cover(v, k, t, rng, deadline):
    """
    Constructs a covering using a greedy heuristic with randomization.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(tuple(uncovered))
        block = list(start_triple)
        
        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                break
                
            # Calculate gain for each candidate point
            best_gain = -1
            best_points = []
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check all pairs in block
                for a, b in itertools.combinations(block, 2):
                    triple = tuple(sorted((a, b, x)))
                    if triple in uncovered:
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
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    Returns the reduced list of blocks.
    """
    if not blocks:
        return blocks
    
    # Map each triple to the list of blocks that cover it
    triple_cover_count = defaultdict(int)
    
    # Build the map
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            # Check if this block is essential
            is_essential = False
            for triple in itertools.combinations(block, t):
                if triple_cover_count[tuple(triple)] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
            else:
                # Remove this block
                changed = True
                # Update cover counts
                for triple in itertools.combinations(block, t):
                    triple_cover_count[tuple(triple)] -= 1
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
    triple_cover_count = defaultdict(int)
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1
    
    best_blocks = blocks[:]
    best_size = len(blocks)
    
    # Try removing non-essential blocks and see if we can re-cover with fewer
    # For now, just do the reduction pass again with different ordering
    for _ in range(5):
        if time.time() > deadline:
            break
        blocks = _reduce_covering(blocks, v, k, t)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
    
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run to have a baseline
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    blocks = _local_search(blocks, v, k, t, rng, deadline)
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
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            if best_size <= 69:
                break
                
    return [list(b) for b in best_blocks]
