import math
import time
from fractions import Fraction

def construct():
    start = time.time()
    DEADLINE = start + 52.0
    
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Compute LCM for exact arithmetic
    D = 1
    for d in denoms:
        D = D * d // math.gcd(D, d)
    
    # w[i] = D // (i+2), so sum of w[i] for selected i = (sum of 1/(i+2)) * D
    w = [D // d for d in denoms]
    
    # Meet-in-the-middle approach
    n1 = n // 2
    n2 = n - n1
    
    # Sort by weight descending to improve pruning
    indexed_w = sorted(enumerate(w), key=lambda x: -x[1])
    sorted_indices = [idx for idx, _ in indexed_w]
    sorted_w = [wt for _, wt in indexed_w]
    
    w1 = sorted_w[:n1]
    w2 = sorted_w[n1:]
    
    # Generate all valid subsets with deduplication by sum value
    # Keep only one mask per distinct sum to reduce memory and search time
    def gen_half_dedup(weights, deadline):
        sums_dict = {0: 0}
        for i in range(len(weights)):
            wi = weights[i]
            mi = 1 << i
            new_sums = dict(sums_dict)
            for s in sums_dict:
                ns = s + wi
                if ns < D and ns not in new_sums:
                    new_sums[ns] = sums_dict[s] | mi
            sums_dict = new_sums
            if time.time() > deadline:
                break
        return list(sums_dict.items())
    
    # First half: generate deduplicated subsets
    packed1 = gen_half_dedup(w1, DEADLINE - 15)
    
    # Second half: generate deduplicated subsets
    packed2 = gen_half_dedup(w2, DEADLINE - 5)
    
    # Sort second half by sum
    packed2.sort(key=lambda x: x[0])
    
    # Extract arrays for efficient binary search
    sums2 = [item[0] for item in packed2]
    masks2 = [item[1] for item in packed2]
    
    # Binary search for best combination
    best_W = -1
    best_mask1 = 0
    best_mask2 = 0
    
    for s1, m1 in packed1:
        if time.time() > DEADLINE - 2:
            break
        target = D - s1 - 1
        lo, hi = 0, len(sums2)
        while lo < hi:
            mid = (lo + hi) // 2
            if sums2[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx >= 0:
            total = s1 + sums2[idx]
            if total > best_W:
                best_W = total
                best_mask1 = m1
                best_mask2 = masks2[idx]
    
    if best_W < 0:
        return [0]
    
    result = []
    for i in range(n1):
        if (best_mask1 >> i) & 1:
            result.append(sorted_indices[i])
    for i in range(n2):
        if (best_mask2 >> i) & 1:
            result.append(sorted_indices[n1 + i])
    
    return result
