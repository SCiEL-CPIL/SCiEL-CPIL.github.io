import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate all 4-subsets of [1, 16] as 16-bit integers
    all_sets = []
    for comb in combinations(range(16), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    # Precompute subsets for all masks
    # subsets_cache[mask] = [list of submasks of size 0, list of size 1, ..., list of size 4]
    subsets_cache = [None] * 65536
    for mask in range(65536):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            bits.append(lsb)
            m ^= lsb
        
        cached_lists = []
        for k in range(5):
            if k > len(bits):
                cached_lists.append([])
            else:
                subs = []
                for comb in combinations(bits, k):
                    sub = 0
                    for b in comb:
                        sub |= b
                    subs.append(sub)
                cached_lists.append(subs)
        subsets_cache[mask] = cached_lists

    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        forbidden = set()
        
        for S in all_sets:
            if S in forbidden:
                continue
            
            # Add S to family
            family.append(S)
            
            # Update forbidden set
            for A in family[:-1]:
                K = S & A
                S_diff = S & ~K
                A_diff = A & ~K
                U = 0xFFFF ^ (S_diff | A_diff)
                base = U & ~K
                k_size = bin(K).count('1')
                need = 4 - k_size
                
                subs = subsets_cache[base][need]
                for sub in subs:
                    C = K | sub
                    forbidden.add(C)
                    
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
    # Convert back to list of sorted ints 1..16
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if (mask >> i) & 1:
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
