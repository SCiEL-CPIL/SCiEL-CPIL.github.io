"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that the signed sum
        S = sum_{i=1}^{30} c[i-1] / i
    is nonzero and has minimal absolute value.
    
    Uses meet-in-the-middle with two-pointer merge to find the minimum
    non-zero absolute sum efficiently.
    """
    n = 30
    # Compute LCM of 1..30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute scaled weights
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2  # 15
    
    # Enumerate all 3^15 sums for the first half
    def enumerate_half(start, length):
        sums = [0]
        for i in range(start, start + length):
            w = weights[i]
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    sums1 = enumerate_half(0, half)
    sums2 = enumerate_half(half, n - half)
    
    sums1.sort()
    sums2.sort()
    
    i = 0
    j = len(sums2) - 1
    best_abs = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    while i < len(sums1) and j >= 0:
        s = sums1[i] + sums2[j]
        
        if s == 0:
            # Check immediate neighbors for non-zero minimum
            if i + 1 < len(sums1):
                s_next = sums1[i + 1] + sums2[j]
                if s_next != 0 and abs(s_next) < best_abs:
                    best_abs = abs(s_next)
                    best_s1 = sums1[i + 1]
                    best_s2 = sums2[j]
                    if best_abs == 1:
                        break
            if j - 1 >= 0:
                s_next = sums1[i] + sums2[j - 1]
                if s_next != 0 and abs(s_next) < best_abs:
                    best_abs = abs(s_next)
                    best_s1 = sums1[i]
                    best_s2 = sums2[j - 1]
                    if best_abs == 1:
                        break
            i += 1
            j -= 1
        else:
            if abs(s) < best_abs:
                best_abs = abs(s)
                best_s1 = sums1[i]
                best_s2 = sums2[j]
                if best_abs == 1:
                    break
            
            if s > 0:
                j -= 1
            else:
                i += 1
                
    # Reconstruct coefficients
    def find_coeffs(start, length, target_sum):
        coeffs = [0] * length
        
        def dfs(pos, current_sum):
            if pos == length:
                return current_sum == target_sum
            
            w = weights[start + pos]
            remaining_max = sum(weights[start + pos + 1 : start + length])
            
            # Pruning
            if abs(target_sum - current_sum) > remaining_max:
                return False
            
            for c in (-1, 0, 1):
                coeffs[pos] = c
                if dfs(pos + 1, current_sum + c * w):
                    return True
            
            coeffs[pos] = 0
            return False
        
        if dfs(0, 0):
            return coeffs
        return None

    coeffs1 = find_coeffs(0, half, best_s1)
    coeffs2 = find_coeffs(half, n - half, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    return coeffs1 + coeffs2

# EVOLVE-BLOCK-END
