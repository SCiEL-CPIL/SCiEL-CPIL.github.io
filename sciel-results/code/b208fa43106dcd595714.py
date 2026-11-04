import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    
    # Run multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, trial)
        if valid and (best_blocks is None or len(blocks) < len(best_blocks)):
            best_blocks = blocks
        
        # Local improvement: try to remove redundant blocks
        if valid:
            improved_blocks = _local_improve(blocks, v, k, t, deadline)
            if len(improved_blocks) < len(blocks):
                blocks = improved_blocks
                if best_blocks is None or len(blocks) < len(best_blocks):
                    best_blocks = blocks
        
        # If we find a very good solution, we might stop early?
        # The known best is 76. If we hit 76, we can stop.
        if best_blocks and len(best_blocks) <= 76:
            break
            
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, -1)
        if valid:
            best_blocks = blocks
            # Try to improve even the fallback
            improved_blocks = _local_improve(best_blocks, v, k, t, deadline)
            if len(improved_blocks) < len(best_blocks):
                best_blocks = improved_blocks
        else:
            # Last resort: return empty list (invalid, but avoids crash)
            best_blocks = []
            
    return best_blocks

def _greedy_cover(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Greedy covering with randomized tie-breaking and ratio-based scoring.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    uncovered = set(all_tsubsets)
    blocks = []
    
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed the block
        seed = rng.choice(list(uncovered))
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            block_set = set(block)
            best_x = -1
            best_score = -1
            best_tiebreakers = []
            
            # Precompute the (t-1)-subsets of the current block
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            for x in points:
                if x in block_set:
                    continue
                
                # Count how many new t-subsets this point would cover
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                
                # Use a ratio-based score: gain / (number of points in block)
                # This penalizes adding points that don't contribute much
                if gain > 0:
                    score = gain / (len(block) + 1)
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_tiebreakers = [x]
                elif score == best_score and score > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                # No point gives positive gain; pick a random one
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                chosen_x = rng.choice(best_tiebreakers)
            
            block.append(chosen_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks, True

def _local_improve(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    """
    if not blocks:
        return blocks
    
    # Build a count for each t-subset: how many blocks cover it
    cover_count = Counter()
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1
    
    n_blocks = len(blocks)
    improved = list(blocks)
    
    # Try removing each block if it's redundant
    # We iterate backwards so that removal doesn't affect indices
    for i in range(n_blocks - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Check if block i is redundant: all its t-subsets are covered by other blocks
        block = improved[i]
        redundant = True
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 1:
                redundant = False
                break
        
        if redundant:
            # Remove block i
            for s in itertools.combinations(block, t):
                cover_count[s] -= 1
            improved.pop(i)
    
    return improved
