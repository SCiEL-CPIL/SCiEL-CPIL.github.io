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
    
    # Search over base-B binary expansion full sets and pruned subsets
    for B in range(2, 20):
        for n in range(2, 11):
            # Generate all base-B binary numbers
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U_full = all_vals
            s_full = calc_score(U_full)
            if s_full > best_score:
                best_score = s_full
                best = U_full
            
            # If the set is small enough, try to improve it by pruning elements
            # Greedy removal: remove element if it increases score
            if len(U_full) <= 32:
                U_curr = U_full[:]
                curr_score = s_full
                improved = True
                while improved:
                    improved = False
                    # Iterate over a copy to allow modification
                    for x in U_curr[:]:
                        U_test = [y for y in U_curr if y != x]
                        if len(U_test) < 2:
                            continue
                        s_test = calc_score(U_test)
                        if s_test > curr_score:
                            U_curr = U_test
                            curr_score = s_test
                            improved = True
                
                if curr_score > best_score:
                    best_score = curr_score
                    best = U_curr
            
            # Also try small subsets for larger n? 
            # If n is large, full set is huge. Pruning is too slow.
            # But maybe specific patterns? 
            # For now, rely on pruning for small n and full sets for large n.
            
    return best
