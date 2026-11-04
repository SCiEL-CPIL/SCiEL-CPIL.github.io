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
    
    # Search over base-B binary expansion full sets
    # U consists of all numbers with digits 0 or 1 in base B, up to n digits.
    
    for B in range(2, 15):
        for n in range(2, 11):
            # Generate all base-B binary numbers
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U = all_vals
            # Ensure 0 is present (it is, mask=0)
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
            
            # Also try subsets of the full set for larger n?
            # For n <= 6, we can try removing elements.
            # But let's first see if full sets are better.
            # If n is small, we can also search subsets as before.
            if n <= 5:
                # Re-run the subset search for this B, n
                for size in range(2, min(6, len(all_vals) + 1)):
                    # To avoid excessive time, only search if 2^n is not too big
                    if len(all_vals) > 32:
                        continue
                    for combo in combinations(all_vals, size):
                        U_sub = [0] + list(combo)
                        U_sub = list(set(U_sub))
                        if len(U_sub) != size + 1:
                            continue
                        s_sub = calc_score(U_sub)
                        if s_sub > best_score:
                            best_score = s_sub
                            best = U_sub

    return best
