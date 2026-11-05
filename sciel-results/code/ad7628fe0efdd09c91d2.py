import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 55
    
    # 1. Generate all 4-subsets
    all_quads = list(combinations(range(1, 17), 4))
    n_quads = len(all_quads)
    masks = [0] * n_quads
    for i, q in enumerate(all_quads):
        m = 0
        for x in q:
            m |= (1 << (x - 1))
        masks[i] = m
        
    # 2. Precompute subsets[W] = list of m such that masks[m] is subset of W
    # W is a 16-bit mask
    subsets = [[] for _ in range(1 << 16)]
    
    # Optimization: Only precompute for W that are complements of U?
    # U is union of two 4-sets minus intersection.
    # Size of U is 4 to 8.
    # Size of W is 8 to 12.
    # We can just compute for all 65536, it's fast enough.
    
    for i, M in enumerate(masks):
        comp = (~M) & 0xFFFF
        sub = comp
        while True:
            V = M | sub
            subsets[V].append(i)
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    best_family = []
    best_size = 0
    
    # Multiple restarts
    num_restarts = 200
    for trial in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle indices for random selection
        indices = list(range(n_quads))
        random.seed(trial)
        random.shuffle(indices)
        
        family = []
        in_family = [False] * n_quads
        block_count = [0] * n_quads
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            if in_family[idx]:
                continue
                
            if block_count[idx] > 0:
                continue
                
            # Add idx to family
            family.append(idx)
            in_family[idx] = True
            
            # Update block counts for all other sets
            # For each i in family (except idx)
            # Find conflicts with pair (i, idx)
            
            K_full = 0xFFFF
            
            for j in family:
                if j == idx:
                    continue
                M_j = masks[j]
                M_idx = masks[idx]
                
                K = M_j & M_idx
                # U = (M_j \ K) | (M_idx \ K)
                # M_j \ K = M_j & ~K
                # M_idx \ K = M_idx & ~K
                # U = (M_j | M_idx) & ~K  <- Wait.
                # (A \ K) | (B \ K) = (A | B) \ K.
                # Yes.
                U = (M_j | M_idx) & ~K
                W = (~U) & 0xFFFF
                
                # Get candidates m such that M_m subset of W
                # And K subset of M_m
                for m in subsets[W]:
                    if (masks[m] & K) == K:
                        block_count[m] += 1
            
            if len(family) > best_size:
                best_size = len(family)
                best_family = family[:]
                
        # End trial
        
    result = [sorted(all_quads[i]) for i in best_family]
    return result
