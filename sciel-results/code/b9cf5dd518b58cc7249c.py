import math
from itertools import combinations

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Define base-expansion universes
    universes = []
    
    # B=2, n=4 (size 16, max 15)
    # B=2, n=5 (size 32, max 31)
    # B=3, n=3 (size 27, max 26)
    
    for B in [2, 3]:
        for n in [3, 4, 5]:
            if B == 3 and n > 3: continue # limit size
            if B == 2 and n > 5: continue
            
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            universes.append(all_vals)
            
    # Search subsets
    for U_univ in universes:
        if 0 not in U_univ:
            continue
        # Remove 0 from pool for combinations, then add back
        pool = [x for x in U_univ if x != 0]
        
        # Determine max subset size to search
        # For small universes, search up to size 8
        # For larger, search up to size 6
        max_k = 8 if len(U_univ) <= 16 else 6
        
        for k in range(3, max_k + 1):
            # We need k-1 elements from pool
            if len(pool) < k - 1:
                continue
            for combo in combinations(pool, k - 1):
                U = [0] + list(combo)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best = U
                    
    return best
