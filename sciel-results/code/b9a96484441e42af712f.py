"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline for 90s limit, 5s safety margin
    
    n = 30
    # Compute LCM(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # weights[i] = L / (i+1) for i in 0..29, so weight for term 1/(i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    # Split into two halves
    mid = n // 2  # 15
    half1 = weights[:mid]   # weights for 1/1, 1/2, ..., 1/15
    half2 = weights[mid:]   # weights for 1/16, 1/17, ..., 1/30
    
    # Enumerate all 3^15 signed sums for each half
    # For each position, coefficient is -1, 0, or +1
    def enumerate_sums(half_weights):
        """Return list of all possible signed sums."""
        sums = [0]
        for w in half_weights:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    # Check time
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums1 = enumerate_sums(half1)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums2 = enumerate_sums(half2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Sort sums2 for binary search
    sums2.sort()
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # For each sum in sums1, find sum in sums2 that makes total closest to 0 but nonzero
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    import bisect
    
    for s1 in sums1:
        # We want s1 + s2 ≈ 0, i.e., s2 ≈ -s1, but s1 + s2 != 0
        target = -s1
        # Binary search for target in sums2
        idx = bisect.bisect_left(sums2, target)
        
        # Check candidates around idx
        candidates = []
        for i in range(max(0, idx - 2), min(len(sums2), idx + 3)):
            total = s1 + sums2[i]
            if total != 0:
                candidates.append((abs(total), total, s1, sums2[i]))
        
        if candidates:
            candidates.sort()
            best_candidate = candidates[0]
            if best_abs is None or best_candidate[0] < best_abs:
                best_abs = best_candidate[0]
                best_s1 = best_candidate[2]
                best_s2 = best_candidate[3]
                if best_abs == 1:  # Can't do better than 1 (in numerator units)
                    break
        
        # Time check every million iterations
        if len(sums1) > 1000000 and (sums1.index(s1) if False else 0) % 1000000 == 0:
            pass  # index is O(n), skip this
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    if best_abs is None:
        # Fallback to seed
        return [0] * 28 + [1, -1]
    
    # Now decode best_s1 and best_s2 back to coefficients
    def decode_sum(half_weights, target_sum):
        """Find coefficients that produce target_sum."""
        n_h = len(half_weights)
        coeffs = [0] * n_h
        # Backtrack
        def backtrack(pos, remaining):
            if pos == n_h:
                return remaining == 0
            w = half_weights[pos]
            for c in (1, 0, -1):
                if backtrack(pos + 1, remaining - c * w):
                    coeffs[pos] = c
                    return True
            return False
        
        if backtrack(0, target_sum):
            return coeffs
        return None
    
    coeffs1 = decode_sum(half1, best_s1)
    coeffs2 = decode_sum(half2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    result = coeffs1 + coeffs2
    
    # Verify
    total = 0
    for i in range(n):
        total += result[i] * weights[i]
    
    if total == 0:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
