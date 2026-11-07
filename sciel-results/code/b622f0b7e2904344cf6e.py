"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from fractions import Fraction
from collections import defaultdict

MAX_N = 10_000_000
TIME_LIMIT = 70  # seconds, safety margin before 90s kill

def compute_sigma_sieve(n):
    """Compute sigma(n) for all n in [1, n] using a vectorized sieve."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        sigma[i::i] += i
    return sigma

def find_best_pair(sigma, time_limit):
    """
    Given sigma array, find pair (a, b) with sigma[a] == sigma[b] 
    that minimizes |a/b - 3/2|, with a, b <= 10^7.
    """
    start_time = time.time()
    
    # Group numbers by their sigma value
    sigma_groups = defaultdict(list)
    for i in range(1, len(sigma)):
        s = int(sigma[i])
        if s > 0:
            sigma_groups[s].append(i)
    
    # Sort each group
    for s in sigma_groups:
        sigma_groups[s].sort()
    
    best_error = Fraction(1, 1)  # Start with a large error
    best_pair = (1, 2)
    
    # For each group with at least 2 elements, find the pair closest to ratio 3/2
    for s, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        
        if time.time() - start_time > time_limit - 5:
            break
        
        # Use binary search for efficiency
        for idx, b in enumerate(nums):
            if time.time() - start_time > time_limit - 5:
                break
            
            # We want a such that a/b ≈ 3/2, so a ≈ 3*b/2
            target_a = 3 * b // 2
            
            # Binary search for target_a in nums
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check candidates around position lo
            candidates = set()
            for pos in [lo - 1, lo, lo + 1]:
                if 0 <= pos < len(nums):
                    candidates.add(nums[pos])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| exactly
                err = abs(Fraction(a, b) - Fraction(3, 2))
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
    
    return best_pair, best_error

def construct():
    start_time = time.time()
    
    # Compute sigma for all numbers up to MAX_N
    sigma = compute_sigma_sieve(MAX_N)
    
    # Find the best pair
    best_pair, best_error = find_best_pair(sigma, TIME_LIMIT)
    
    a, b = best_pair
    
    # Ensure they are Python ints
    a = int(a)
    b = int(b)
    
    # Check that error is not exactly 0
    if best_error == 0:
        # Find a nearby pair with non-zero error
        # Just return a hardcoded fallback
        a, b = 11, 6
    
    return [a, b]
# EVOLVE-BLOCK-END
