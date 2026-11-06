import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for the 120s limit
    
    points = list(range(12))
    all_blocks = list(combinations(points, 6))
    
    # Precompute 5-subsets for each block to speed up checking
    block_5subs = []
    for b in all_blocks:
        block_5subs.append(tuple(tuple(c) for c in combinations(b, 5)))
        
    best_blocks = []
    best_count = 0
    
    # We know the max is 132. 
    # If we find 132, we can stop.
    
    while time.time() < deadline:
        # Random restart
        shuffled_blocks = all_blocks[:]
        random.shuffle(shuffled_blocks)
        
        covered = set()
        current_blocks = []
        
        for i, b in enumerate(shuffled_blocks):
            # Check if any 5-subset is already covered
            valid = True
            for sub in block_5subs[i]:
                if sub in covered:
                    valid = False
                    break
            
            if valid:
                current_blocks.append(list(b))
                covered.update(block_5subs[i])
                
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
            if best_count == 132:
                # Optimum reached
                return best_blocks

    return best_blocks
