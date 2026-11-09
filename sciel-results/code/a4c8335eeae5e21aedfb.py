"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that the signed sum
        S = sum_{i=1}^{30} c[i-1] / i
    is nonzero and has minimal absolute value.
    
    Uses meet-in-the-middle:
    - Split indices into two halves of 15.
    - Enumerate all 3^15 = 14348901 combinations for each half.
    - For each half, compute the exact rational sum (numerator, denominator)
      scaled to a common denominator (LCM of 1..30).
    - Sort one half's sums.
    - For each sum in the other half, binary search for the complement that
      minimizes |total|, excluding zero.
    """
    n = 30
    # Compute LCM of 1..30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute scaled weights: w[i] = L / (i+1) for i in 0..29
    # So sum c[i] * w[i] = L * S
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2  # 15
    # First half: indices 0..14
    # Second half: indices 15..29
    
    # Enumerate all 3^15 sums for the first half
    def enumerate_half(start, length):
        """Enumerate all possible signed sums for weights[start:start+length]."""
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
    
    # This can be memory-intensive: 3^15 = 14348901 integers each
    # Let's generate them efficiently
    
    sums1 = enumerate_half(0, half)
    sums2 = enumerate_half(half, n - half)
    
    # Sort sums2 for binary search
    sums2.sort()
    
    # For each sum in sums1, we want to find sum in sums2 such that
    # |sum1 + sum2| is minimized and nonzero.
    # Since total_scaled = sum1 + sum2, and S = total_scaled / L,
    # minimizing |S| is equivalent to minimizing |total_scaled|.
    
    from bisect import bisect_left, bisect_right
    
    best_abs = None
    best_c1 = None
    best_c2 = None
    
    # We need to recover the coefficients. Let's store them.
    # Actually, storing all coefficient vectors is too much memory.
    # Alternative: store (sum, coefficients) but that's heavy.
    
    # Better approach: for each half, store a list of (sum, coeff_tuple).
    # But 3^15 tuples of 15 ints each is too much.
    
    # Alternative: just find the optimal total_scaled value first,
    # then reconstruct coefficients by backtracking.
    
    # Let's first find the minimal nonzero |total_scaled|.
    
    best_val = None
    
    for s1 in sums1:
        # We want s2 such that |s1 + s2| is minimized and s1 + s2 != 0
        # So we want s2 near -s1
        target = -s1
        
        # Find insertion point in sums2
        idx = bisect_left(sums2, target)
        
        # Check candidates around idx
        candidates = []
        for i in (idx - 1, idx, idx + 1):
            if 0 <= i < len(sums2):
                candidates.append(sums2[i])
        
        # Also, we might need to check more if there are duplicates
        # But bisect gives us the right neighborhood
        
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if best_val is None or abs_total < best_val:
                best_val = abs_total
                best_s1 = s1
                best_s2 = s2
    
    # Now we need to find coefficients that achieve best_s1 and best_s2.
    # We can do this by dynamic programming or backtracking.
    
    def find_coeffs(start, length, target_sum):
        """Find coefficients in {-1,0,1} for weights[start:start+length] that sum to target_sum."""
        # Use DFS with pruning
        coeffs = [0] * length
        
        def dfs(pos, current_sum):
            if pos == length:
                if current_sum == target_sum:
                    return True
                return False
            
            w = weights[start + pos]
            
            # Try -1, 0, +1
            for c in (-1, 0, 1):
                coeffs[pos] = c
                new_sum = current_sum + c * w
                # Pruning: check if it's possible to reach target
                # Remaining max contribution: sum of remaining weights
                remaining_max = sum(weights[start + pos + 1 : start + length])
                if abs(target_sum - new_sum) <= remaining_max:
                    if dfs(pos + 1, new_sum):
                        return True
            
            coeffs[pos] = 0
            return False
        
        if dfs(0, 0):
            return coeffs
        return None
    
    coeffs1 = find_coeffs(0, half, best_s1)
    coeffs2 = find_coeffs(half, n - half, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        # Fallback: return the seed solution
        return [0] * 28 + [1, -1]
    
    return coeffs1 + coeffs2

# EVOLVE-BLOCK-END
