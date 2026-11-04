import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate all 6-subsets of indices 0..10
    all_subsets = list(itertools.combinations(range(D), 6))
    
    best_blocks = []
    
    # Try multiple random orderings to find a large packing
    # A packing is a set of subsets where any two intersect in at most 3 elements.
    # This is equivalent to no 4-subset being contained in more than one block.
    
    for _ in range(1000):
        random.shuffle(all_subsets)
        current_blocks = []
        covered_4s = set()
        
        for block in all_subsets:
            block_4s = list(itertools.combinations(block, 4))
            valid = True
            for t in block_4s:
                if t in covered_4s:
                    valid = False
                    break
            
            if valid:
                current_blocks.append(block)
                for t in block_4s:
                    covered_4s.add(t)
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) >= 22:
                break
    
    # Construct vectors
    vectors = []
    
    # All 64 sign combinations for 6 dimensions
    all_signs = list(itertools.product([1, -1], repeat=6))
    
    for block in best_blocks:
        for signs in all_signs:
            v = [0] * D
            for idx, sign in zip(block, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
