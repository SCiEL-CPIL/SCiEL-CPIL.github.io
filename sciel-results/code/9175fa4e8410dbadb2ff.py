import itertools
import time
import random
from collections import defaultdict

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
        # If we find a very good solution, we might stop early?
        # The known best is 76. If we hit 76, we can stop.
        if best_blocks and len(best_blocks) <= 76:
            break
            
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        # even if the random trials failed or timed out badly.
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, -1)
        if valid:
            best_blocks = blocks
        else:
            # Last resort: return empty list (invalid, but avoids crash)
            best_blocks = []
            
    return best_blocks

def _greedy_cover(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Greedy covering with randomized tie-breaking.
    If seed_val >= 0, use it to seed the random number generator for reproducibility within the trial.
    """
    # Use a local random instance to avoid global state issues
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    uncovered = set(all_tsubsets)
    blocks = []
    
    # Precompute the list of all points for faster iteration
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed
        # Using a list conversion each time is O(n), but n is small (2024 max)
        # Alternatively, we could maintain a list of uncovered, but set is easier for removal.
        # For small sets, random.choice(list(uncovered)) is fine.
        seed = rng.choice(list(uncovered))
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            block_set = set(block)
            best_x = -1
            best_gain = -1
            best_tiebreakers = []
            
            # Precompute (t-1) subsets of block to speed up gain calculation
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            # Calculate gain for each candidate
            for x in points:
                if x in block_set:
                    continue
                gain = 0
                for p in t_minus_1_subsets:
                    # Create the t-subset p U {x}
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tiebreakers = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                # Should not happen if uncovered is not empty and we can add points
                # But if no point covers anything new, we are stuck.
                return None, False
                
            # Randomly select from the best candidates to break ties
            chosen_x = rng.choice(best_tiebreakers)
            block.append(chosen_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks, True
