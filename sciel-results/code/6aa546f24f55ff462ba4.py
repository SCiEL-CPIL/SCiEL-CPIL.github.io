import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    # Precompute 5-subsets for each block
    block_subsets = [set(combinations(b, 5)) for b in all_blocks]
    
    best_blocks = []
    best_count = 0
    
    # Indices for shuffling
    indices = list(range(len(all_blocks)))
    
    while time.time() < deadline:
        random.shuffle(indices)
        covered = set()
        current_blocks = []
        
        for i in indices:
            b = all_blocks[i]
            subs = block_subsets[i]
            # Check intersection
            # Using set intersection is fast
            if not covered & subs:
                current_blocks.append(b)
                covered |= subs
                
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    return [list(b) for b in best_blocks]
