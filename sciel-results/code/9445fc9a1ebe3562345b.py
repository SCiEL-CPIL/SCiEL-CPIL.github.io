"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}
    
    # subsets_by_point[x] = list of indices of subsets containing x
    subsets_by_point = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for x in s:
            subsets_by_point[x].append(i)
    
    covered = np.zeros(n_subsets, dtype=bool)
    blocks = []
    
    # Helper to mark subsets of a block as covered
    def mark_covered(block):
        for s in itertools.combinations(block, t):
            idx = subset_to_index[s]
            covered[idx] = True

    while True:
        if time.time() > deadline:
            break
            
        # Find first uncovered subset
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
            
        first_uncovered_idx = uncovered_indices[0]
        first_uncovered_subset = all_subsets[first_uncovered_idx]
        
        block = set(first_uncovered_subset)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                    
                gain = 0
                # Check subsets containing x
                for idx in subsets_by_point[x]:
                    if covered[idx]:
                        continue
                    s = all_subsets[idx]
                    # Check if s \ {x} is subset of block
                    # s is a tuple of t elements
                    # We need to check if all elements in s except x are in block
                    # Since x is in s, we can just check the other t-1 elements
                    # This is a bit slow in Python, let's optimize
                    # For t=3, s=(a,b,c). If x==a, check b,c in block.
                    is_subset = True
                    for elem in s:
                        if elem == x:
                            continue
                        if elem not in block:
                            is_subset = False
                            break
                    if is_subset:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1:
                # Fallback, should not happen
                break
                
            block.add(best_x)
            
        block_list = sorted(block)
        blocks.append(block_list)
        mark_covered(block_list)
        
    return blocks
# EVOLVE-BLOCK-END
