import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    all_ones = (1 << n) - 1
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    best_family = []
    best_size = 0
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    while time.time() < deadline:
        random.shuffle(all_sets)
        current_family = []
        current_set = set()
        
        for A in all_sets:
            conflict = False
            for B in current_family:
                K = A & B
                k_size = bin(K).count('1')
                M = (A | B) ^ K
                R = all_ones ^ (K | M)
                
                need = k - k_size
                if bin(R).count('1') < need:
                    continue
                
                # Extract bits from R
                bits_in_R = []
                r = R
                while r:
                    lsb = r & -r
                    bits_in_R.append(lsb)
                    r ^= lsb
                
                if len(bits_in_R) < need:
                    continue
                    
                for p_comb in combinations(bits_in_R, need):
                    P = 0
                    for bit in p_comb:
                        P |= bit
                    C = K | P
                    if C in current_set:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                current_family.append(A)
                current_set.add(A)
                if len(current_family) >= 250:
                    break

        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            if best_size >= 250:
                break
        
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
