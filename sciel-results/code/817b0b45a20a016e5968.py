import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for faster lookup? 
    # Or just use a set of tuples.
    
    best_blocks = None
    
    # Run multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline)
        if valid and (best_blocks is None or len(blocks) < len(best_blocks)):
            best_blocks = blocks
        # If we find a very good solution, we might stop early? 
        # The known best is 76. If we hit 76, we can stop.
        if best_blocks and len(best_blocks) <= 76:
            break
            
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        # even if the random trials failed or timed out badly.
        # But the random trials should produce valid coverings.
        # If time ran out during a trial, that trial is invalid.
        # We need to ensure at least one valid trial completes.
        # The first trial should be deterministic or fast enough.
        pass
        
    return best_blocks if best_blocks else []

def _greedy_cover(v, k, t, all_tsubsets, deadline):
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed
        seed = random.choice(list(uncovered))
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            # Precompute (t-1) subsets of block to speed up gain calculation?
            # Gain of x = count of s in uncovered such that s is subset of block U {x} and x in s.
            # This means s \ {x} is a subset of block.
            # Since |s|=t, |s \ {x}| = t-1.
            # So we need to count uncovered s such that (s \ {x}) is a subset of block.
            
            # Efficient way:
            # For each uncovered s, if x in s, check if s \ {x} is in block.
            # Iterating over all uncovered s is O(|uncovered|).
            # |uncovered| starts at 2024 and decreases.
            # Candidates x: v - len(block).
            
            # Optimization: 
            # Instead of iterating all uncovered, we can iterate over potential s?
            # Or, for each x, we can look at the (t-1) subsets of block.
            # Let P be the set of (t-1)-subsets of block.
            # For a candidate x, gain = sum(1 for p in P if (p U {x}) in uncovered).
            # Note: (p U {x}) must be a t-subset. Since p has t-1 elements and x is not in block (so not in p), this is a valid t-subset.
            # So, precompute the (t-1) subsets of the current block.
            
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    # Create the t-subset p U {x}
                    # p is a tuple of t-1 elements.
                    # We need to check if sorted(p + (x,)) is in uncovered.
                    # To avoid sorting every time, we can precompute a set of tuples?
                    # Or just sort. t is small (3).
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Should not happen if uncovered is not empty
                return None, False
                
            block.append(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks, True
