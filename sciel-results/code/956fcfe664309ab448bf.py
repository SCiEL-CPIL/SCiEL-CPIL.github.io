"""Program for small nonzero signed harmonic subsums."""
import time
import numpy as np
from fractions import Fraction
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    Uses NumPy-accelerated Meet-in-the-Middle.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin

    n = 30

    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)

    # weights[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    # Check if L fits in int64
    # L ~ 2.3e12, int64 max ~ 9.2e18. Yes.
    
    # Split into two halves
    half = n // 2
    left_weights = weights[:half]
    right_weights = weights[half:]

    # Enumerate sums for left half
    # We use a list for generation, then convert to NumPy for sorting
    left_sums = [0]
    for w in left_weights:
        # Optimization: list comprehension is faster than append loop
        # But we need to triple the list.
        # s, s+w, s-w
        # Using list comprehension with a flat generator
        left_sums = [s + delta for s in left_sums for delta in (0, w, -w)]
        
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Enumerate sums for right half
    right_sums = [0]
    for w in right_weights:
        right_sums = [s + delta for s in right_sums for delta in (0, w, -w)]
        
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Convert to NumPy int64 arrays for fast sorting
    left_arr = np.array(left_sums, dtype=np.int64)
    right_arr = np.array(right_sums, dtype=np.int64)
    
    # Sort
    left_arr.sort()
    right_arr.sort()
    
    # Two-pointer search
    # We want min non-zero |l + r|
    i = 0
    j = len(right_arr) - 1
    best_int_sum = None
    best_left_sum = None
    best_right_sum = None
    
    # To speed up the loop, we can access NumPy elements directly
    # But Python loop over 30M iterations might be slow.
    # However, 30M simple iterations in Python is ~1-2s.
    
    while i < len(left_arr) and j >= 0:
        if time.time() > deadline:
            break
            
        l_val = left_arr[i]
        r_val = right_arr[j]
        current_sum = l_val + r_val
        
        if current_sum != 0:
            abs_sum = abs(current_sum)
            if best_int_sum is None or abs_sum < best_int_sum:
                best_int_sum = abs_sum
                best_left_sum = l_val
                best_right_sum = r_val
                
                if best_int_sum == 1:
                    break
            
        if current_sum > 0:
            j -= 1
        else:
            i += 1
            
    if best_int_sum is None:
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    def reconstruct(weights, target):
        """Backtracking to find coefficients in {-1,0,1} summing to target."""
        coeffs = [0] * len(weights)
        
        # Precompute suffix sums for pruning
        suffix_sums = [0] * (len(weights) + 1)
        for i in range(len(weights) - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + abs(weights[i])
            
        def bt(idx, cur_sum):
            if idx == len(weights):
                return cur_sum == target
            
            # Pruning: if the maximum possible change from remaining weights
            # cannot bridge the gap to target, return False
            if abs(target - cur_sum) > suffix_sums[idx]:
                return False
                
            w = weights[idx]
            # Try values in order of likelihood? 0 is often best for small sums?
            # Or just 0, 1, -1
            for val in (0, 1, -1):
                coeffs[idx] = val
                if bt(idx + 1, cur_sum + val * w):
                    return True
            coeffs[idx] = 0
            return False
        
        if bt(0, 0):
            return coeffs
        return None

    left_coeffs = reconstruct(left_weights, int(best_left_sum))
    right_coeffs = reconstruct(right_weights, int(best_right_sum))

    if left_coeffs is None or right_coeffs is None:
        return [0] * 28 + [1, -1]

    result = left_coeffs + right_coeffs

    # Verify
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(result) if c != 0)
    if total_frac == 0:
        return [0] * 28 + [1, -1]

    return result

# EVOLVE-BLOCK-END
