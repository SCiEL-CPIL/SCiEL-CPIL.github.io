import math
import time
import heapq
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
    
    # Generate all valid subsets for first half
    def gen_half(weights, deadline):
        packed = [0]
        for i in range(len(weights)):
            wi = weights[i]
            mi = 1 << i
            new_packed = []
            for p in packed:
                s = p
                if s + wi < D:
                    new_packed.append((s + wi, mi))
                new_packed.append((s, 0))
            packed = new_packed
            if time.time() > deadline:
                break
        return packed
    
    # First half: generate all subsets with sum < D
    packed1 = [(0, 0)]
    for i in range(n1):
        wi = w1[i]
        mi = 1 << i
        new_packed1 = []
        for s, m in packed1:
            if s + wi < D:
                new_packed1.append((s + wi, m | mi))
            new_packed1.append((s, m))
        packed1 = new_packed1
        if time.time() > DEADLINE - 15:
            break
    
    # Second half: generate all subsets with sum < D
    packed2 = [(0, 0)]
    for i in range(n2):
        wi = w2[i]
        mi = 1 << i
        new_packed2 = []
        for s, m in packed2:
            if s + wi < D:
                new_packed2.append((s + wi, m | mi))
            new_packed2.append((s, m))
        packed2 = new_packed2
        if time.time() > DEADLINE - 5:
            break
    
    # Sort second half by sum
    packed2.sort(key=lambda x: x[0])
    
    # Binary search for best combination
    best_W = -1
    best_mask1 = 0
    best_mask2 = 0
    
    for s1, m1 in packed1:
        if time.time() > DEADLINE - 2:
            break
        # Find largest s2 in packed2 such that s1 + s2 < D
        target = D - s1 - 1  # We need s2 <= target
        lo, hi = 0, len(packed2)
        while lo < hi:
            mid = (lo + hi) // 2
            if packed2[mid][0] <= target:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx >= 0:
            s2, m2 = packed2[idx]
            total = s1 + s2
            if total > best_W:
                best_W = total
                best_mask1 = m1
                best_mask2 = m2
    
    if best_W < 0:
        return [0]
    
    # Map back to original indices
    result = []
    for i in range(n1):
        if (best_mask1 >> i) & 1:
            result.append(sorted_indices[i])
    for i in range(n2):
        if (best_mask2 >> i) & 1:
            result.append(sorted_indices[n1 + i])
    
    return result
