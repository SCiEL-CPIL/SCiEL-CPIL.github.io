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
    
    # Sort both sums for Two-Pointer approach
    sums1.sort()
    sums2.sort()
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Two-pointer search for closest non-zero sum
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    i = 0
    j = len(sums2) - 1
    n1 = len(sums1)
    n2 = len(sums2)
    
    while i < n1 and j >= 0:
        if time.time() > deadline:
            break
            
        total = sums1[i] + sums2[j]
        abs_total = abs(total)
        
        if total != 0:
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_s1 = sums1[i]
                best_s2 = sums2[j]
                if best_abs == 1:
                    break
        
        if total < 0:
            i += 1
        elif total > 0:
            j -= 1
        else:
            # total == 0, we need non-zero, so move both to explore neighbors
            i += 1
            j -= 1
    
    if time.time() > deadline and best_abs is None:
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
