import itertools
import time
import random
from collections import defaultdict

def _greedy_cover(v, k, t, rng):
    """
    Constructs a covering using a greedy heuristic with randomization.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        
        # Greedily add points to the block
        while len(block) < k:
            # Calculate gain for each candidate point
            best_gain = -1
            best_points = []
            block_set = set(block)
            
            # Pre-calculate pairs in current block for faster gain calculation
            # Gain of x is number of pairs {a,b} in block such that {a,b,x} in uncovered
            # We can iterate over pairs in block
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check all pairs in block
                for a, b in itertools.combinations(block, 2):
                    # Create the triple {a, b, x}
                    # We need to check if this triple is in uncovered
                    # To make it a hashable key, we sort it
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
                    # Should not happen if v > k
                    break
            else:
                x = rng.choice(best_points)
            
            block.append(x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    Returns the reduced list of blocks.
    """
    if not blocks:
        return blocks
    
    # Map each triple to the list of blocks that cover it
    # Or just count how many blocks cover each triple
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

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run to have a baseline
    # Using a fixed seed for reproducibility of the first attempt
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # Optional: Vary the greedy strategy slightly
        # For now, just randomizing the choice among best gains is enough
        
        blocks = _greedy_cover(v, k, t, rng)
        
        # Check time before reduction, as it can be slower
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a known good value, we could stop
            # Lower bound for C(24,7,3) is 69. If we get 69, we can stop.
            if best_size <= 69:
                break
                
    return best_blocks
