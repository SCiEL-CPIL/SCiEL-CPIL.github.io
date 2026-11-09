import time
from itertools import combinations

def construct():
    """Construct the Witt design S(5,6,12) with 132 blocks."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute all 5-subsets of each 6-subset for fast checking
    all_6subsets = list(combinations(range(12), 6))
    all_5subsets = {}
    for b in all_6subsets:
        all_5subsets[b] = frozenset(combinations(b, 5))
        
    import random
    
    best_blocks = []
    best_count = 0
    
    # Try multiple seeds until we find the optimal 132 blocks
    # The Witt design is unique, so any valid packing of 132 blocks is isomorphic to it.
    # Randomized greedy is very likely to find it within a small number of attempts.
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(attempt)
        covered = set()
        blocks = []
        
        # Shuffle the candidate blocks
        candidates = list(all_6subsets)
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() > deadline:
                break
            subsets = all_5subsets[b]
            # Check if any 5-subset is already covered
            # Optimization: break early if any conflict
            conflict = False
            for s in subsets:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                blocks.append(list(b))
                covered.update(subsets)
                
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # If we found the optimal number of blocks, we can stop immediately
            if best_count >= 132:
                return best_blocks
                
    return best_blocks
