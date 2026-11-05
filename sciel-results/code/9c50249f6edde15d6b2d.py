import time
from fractions import Fraction
from math import gcd
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s wall-clock limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Compute LCM(2, 3, ..., 48)
    L = 1
    for d in denominators:
        L = L // gcd(L, d) * d

    # Integer weights: w_i = L / d_i
    w = [L // d for d in denominators]

    # Meet-in-the-middle
    mid = n // 2  # 23
    n1 = mid
    n2 = n - mid  # 24

    # Generate all valid subsets (sum < L) for each half
    # We return list of (sum, mask)
    def gen_valid_subsets(weights, limit):
        results = [(0, 0)]
        for i, wt in enumerate(weights):
            bit = 1 << i
            # Optimization: only add if it doesn't exceed limit
            # But we need to keep existing ones
            new_entries = []
            # We can optimize by iterating in reverse or using a list of tuples
            # Current approach is O(2^k) which is fine for k=23
            for s, m in results:
                ns = s + wt
                if ns < limit:
                    new_entries.append((ns, m | bit))
            results.extend(new_entries)
        return results

    # Generate right subsets first
    right_subsets = gen_valid_subsets(w[n1:], L)
    
    if time.time() > deadline - 5:
        # Fallback: simple greedy
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Sort right subsets by sum
    right_subsets.sort(key=lambda x: x[0])
    right_sums = [x[0] for x in right_subsets]
    right_masks = [x[1] for x in right_subsets]
    max_right_sum = right_sums[-1] if right_sums else 0

    # Generate left subsets
    left_subsets = gen_valid_subsets(w[:n1], L)
    
    if time.time() > deadline - 5:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Sort left subsets by sum DESCENDING to allow early termination
    left_subsets.sort(key=lambda x: x[0], reverse=True)

    best_total = -1
    best_left_mask = 0
    best_right_mask = 0

    # Iterate left subsets
    for ls, lm in left_subsets:
        # Early termination check
        # If even adding the max possible right sum doesn't beat current best, break
        if ls + max_right_sum <= best_total:
            break
            
        if time.time() > deadline - 3:
            break
        
        # Need rs < L - ls
        bound = L - ls
        # Find rightmost index where right_sums[idx] < bound
        idx = bisect.bisect_left(right_sums, bound) - 1
        if idx < 0:
            continue
        
        rs = right_sums[idx]
        rm = right_masks[idx]
        total = ls + rs
        
        # Total must be strictly less than L
        if total > best_total and (lm | rm) != 0:
            best_total = total
            best_left_mask = lm
            best_right_mask = rm

    # Reconstruct selected indices
    selected = []
    for i in range(n1):
        if best_left_mask & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_right_mask & (1 << i):
            selected.append(n1 + i)

    if not selected:
        # Fallback greedy
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Exact verification with Fraction
    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
    
    if total_frac >= 1:
        # Fallback
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    return selected
