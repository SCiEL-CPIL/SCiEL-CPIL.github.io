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

    # Split into two halves
    mid = n // 2  # 23
    n1 = mid
    n2 = n - mid  # 24

    # Generate all valid subsets (sum < L) for each half
    def gen_valid_subsets(weights, limit):
        """Generate all subsets with sum < limit. Returns list of (sum, mask)."""
        results = [(0, 0)]  # empty set
        for i, wt in enumerate(weights):
            bit = 1 << i
            # For each existing subset, try adding weight i
            new_entries = []
            for s, m in results:
                ns = s + wt
                if ns < limit:  # strictly below 1
                    new_entries.append((ns, m | bit))
            results.extend(new_entries)
        return results

    # Generate for left half
    left_subsets = gen_valid_subsets(w[:n1], L)
    
    if time.time() > deadline - 5:
        # Fallback: simple greedy
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Generate for right half
    right_subsets = gen_valid_subsets(w[n1:], L)
    
    if time.time() > deadline - 5:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Sort right subsets by sum for binary search
    right_sorted = sorted(right_subsets, key=lambda x: x[0])
    right_sums = [x[0] for x in right_sorted]
    right_masks = [x[1] for x in right_sorted]

    # For each left subset, find the best right subset such that total < L
    best_total = -1
    best_left_mask = 0
    best_right_mask = 0

    for ls, lm in left_subsets:
        if time.time() > deadline - 3:
            break
        
        # Need rs < L - ls
        bound = L - ls
        # Find rightmost index where right_sums[idx] < bound
        # bisect_left gives first index where right_sums[idx] >= bound
        idx = bisect.bisect_left(right_sums, bound) - 1
        if idx < 0:
            continue
        
        rs = right_sums[idx]
        rm = right_masks[idx]
        total = ls + rs
        
        # Total must be strictly less than L (guaranteed by construction)
        # and non-empty overall
        if total > best_total and (lm | rm) != 0:
            best_total = total
            best_left_mask = lm
            best_right_mask = rm

    # Also consider: maybe a single half alone is better
    # Left alone
    for ls, lm in left_subsets:
        if lm != 0 and ls > best_total:
            best_total = ls
            best_left_mask = lm
            best_right_mask = 0
    # Right alone
    for rs, rm in right_subsets:
        if rm != 0 and rs > best_total:
            best_total = rs
            best_left_mask = 0
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
        # Should not happen, but fallback
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    return selected
