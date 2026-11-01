import itertools
import time
import random
from collections import defaultdict

def _greedy_random(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    
    # Pre-compute all t-subsets and map to indices for fast lookup?
    # Or just use a set of tuples for uncovered.
    # Set of tuples is fine for v=24. C(24,3)=2024.
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    if not uncovered:
        return []

    while uncovered:
        # Pick a random uncovered triple to start the block
        # To ensure we make progress, we must pick from uncovered.
        # Random choice from a set is O(n) in Python if not careful, 
        # but for 2024 elements it's fine. 
        # Better: convert to list once? No, it changes.
        # rng.choice(list(uncovered)) is O(|uncovered|).
        # This is acceptable.
        
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: how many uncovered triples are covered by adding x
                # A triple is covered if it contains x and t-1 points from block
                # t=3, so we look for pairs in block.
                gain = 0
                if t == 3:
                    # Pairs in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = tuple(sorted((block[i], block[j], x)))
                            if tri in uncovered:
                                gain += 1
                else:
                    # General case
                    for comb in itertools.combinations(block, t-1):
                        tri = tuple(sorted(list(comb) + [x]))
                        if tri in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            # Pick a random candidate from the best ones
            if candidates:
                next_point = rng.choice(candidates)
            else:
                # Fallback: pick any point not in block
                next_point = rng.choice([x for x in all_points if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for tri in itertools.combinations(block, t):
            uncovered.discard(tuple(tri))
            
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        blocks = _greedy_random(v, k, t, seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we found a very good solution, we might want to spend time on local search?
        # For now, just keep restarting.
        # We can stop early if we hit a theoretical lower bound?
        # Lower bound for C(24,7,3) is 69.
        if best_len == 69:
            break
            
        seed += 1
        
    return best_blocks
