import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 55
    
    all_quads = list(combinations(range(1, 17), 4))
    n_quads = len(all_quads)
    masks = [0] * n_quads
    for i, q in enumerate(all_quads):
        m = 0
        for x in q:
            m |= (1 << (x - 1))
        masks[i] = m
        
    # Precompute subsets[W] for all W
    subsets = [[] for _ in range(1 << 16)]
    for i, M in enumerate(masks):
        comp = (~M) & 0xFFFF
        sub = comp
        while True:
            V = M | sub
            subsets[V].append(i)
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    # Precompute candidates[K] for K that are intersections of two 4-sets
    # K is a mask.
    # We only need K with popcount 2, 3, 4.
    # For popcount 0, 1, subsets[W] is better.
    candidates = [[] for _ in range(1 << 16)]
    for i, M in enumerate(masks):
        # M is a 4-set.
        # It contains K if K is a subset of M.
        # We want to add i to candidates[K] for all K subset of M with |K| >= 2.
        # Subsets of M of size 2: 6
        # Size 3: 4
        # Size 4: 1
        # Total 11 subsets.
        # We can iterate them.
        # But we need the mask of the subset.
        # Let's generate all subsets of M.
        sub = M
        while True:
            pop = bin(sub).count('1')
            if pop >= 2:
                candidates[sub].append(i)
            if sub == 0:
                break
            sub = (sub - 1) & M
            
    best_family = []
    best_size = 0
    
    num_restarts = 500
    for trial in range(num_restarts):
        if time.time() > deadline:
            break
            
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
                
            family.append(idx)
            in_family[idx] = True
            
            M_idx = masks[idx]
            
            for j in family:
                if j == idx:
                    continue
                M_j = masks[j]
                
                K = M_j & M_idx
                U = (M_j | M_idx) & ~K
                W = (~U) & 0xFFFF
                
                pop_k = bin(K).count('1')
                
                if pop_k >= 2:
                    for m in candidates[K]:
                        if (masks[m] | W) == W:
                            block_count[m] += 1
                else:
                    for m in subsets[W]:
                        if (masks[m] & K) == K:
                            block_count[m] += 1
                            
            if len(family) > best_size:
                best_size = len(family)
                best_family = family[:]
                
    result = [sorted(all_quads[i]) for i in best_family]
    return result
