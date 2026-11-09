import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    timeout = 80  # Safety margin
    
    n = 30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate sums for first half
    # 3^15 = 14348901
    # Using numpy to generate efficiently
    def gen_sums(w):
        sums = np.array([0], dtype=np.int64)
        for wi in w:
            # Create three copies and add offsets
            # This creates intermediate arrays, but numpy is fast
            sums = np.concatenate((sums - wi, sums, sums + wi))
        return sums

    # Check time
    if time.time() - start_time > timeout * 0.2:
         # This shouldn't happen, but just in case
         pass

    try:
        sums1 = gen_sums(w1)
        sums2 = gen_sums(w2)
        
        # Sort sums2 for binary search
        sums2.sort()
        
        # Find best pair
        best_abs = None
        best_s1 = 0
        best_s2 = 0
        
        # Iterate through sums1
        # We can do this in a loop. 14M iterations in Python is slow (~10-20s).
        # Can we vectorize the search?
        # np.searchsorted works on arrays.
        # targets = -sums1
        # indices = np.searchsorted(sums2, targets)
        # Then we need to check indices and indices-1.
        
        targets = -sums1
        indices = np.searchsorted(sums2, targets)
        
        # Clip indices to valid range [0, len(sums2)-1]
        # But we need to check i-1 and i.
        # So we need to handle boundaries.
        
        # Let's create arrays of candidate s2 values.
        # Candidate 1: sums2[indices]
        # Candidate 2: sums2[indices - 1]
        
        idx1 = indices
        idx2 = indices - 1
        
        # Mask valid indices
        mask1 = (idx1 >= 0) & (idx1 < len(sums2))
        mask2 = (idx2 >= 0) & (idx2 < len(sums2))
        
        # Initialize best candidates
        # We want to minimize |sums1 + s2| where s2 is from sums2.
        
        # For each i, s2_candidate_1 = sums2[idx1[i]]
        # s2_candidate_2 = sums2[idx2[i]]
        
        # We must exclude sum1 + s2 == 0.
        
        # Let's compute the absolute differences for valid candidates.
        # diff1 = |sums1 + sums2[idx1]|
        # diff2 = |sums1 + sums2[idx2]|
        
        # Initialize with infinity
        diff1 = np.full(len(sums1), np.iinfo(np.int64).max, dtype=np.int64)
        diff2 = np.full(len(sums1), np.iinfo(np.int64).max, dtype=np.int64)
        
        # Compute for valid idx1
        if np.any(mask1):
            s2_cand1 = sums2[idx1[mask1]]
            s1_cand1 = sums1[mask1]
            total1 = s1_cand1 + s2_cand1
            # Exclude zero
            valid1 = total1 != 0
            abs_total1 = np.abs(total1[valid1])
            
            # Map back to original indices?
            # This is getting complicated to map back to a single global best.
            
            # Simpler: Just iterate in Python but use the precomputed indices?
            # Or just do the binary search in Python.
            # 14M iterations of bisect in Python is ~15-30s.
            # Given 60s limit, it's risky but possible.
            
            # Let's try the vectorized approach properly.
            
            # We want the global minimum of |s1 + s2| over all valid pairs.
            
            # Let's create an array of all potential (s1, s2) pairs that are close.
            # Actually, the vectorized diff calculation is better.
            
            # Let's re-evaluate.
            # We have arrays sums1, sums2.
            # We want min |s1 + s2| != 0.
            
            # For each s1, the best s2 is either at idx or idx-1.
            
            # Let's collect all candidate s2 values into two arrays.
            
            cand1_s2 = np.zeros(len(sums1), dtype=np.int64)
            cand2_s2 = np.zeros(len(sums1), dtype=np.int64)
            
            # Default to 0, but we will mask them out.
            
            # Fill cand1_s2 where mask1 is true
            cand1_s2[mask1] = sums2[idx1[mask1]]
            # Fill cand2_s2 where mask2 is true
            cand2_s2[mask2] = sums2[idx2[mask2]]
            
            # Calculate totals
            total1 = sums1 + cand1_s2
            total2 = sums1 + cand2_s2
            
            # Mask out invalid indices and zero sums
            valid1 = mask1 & (total1 != 0)
            valid2 = mask2 & (total2 != 0)
            
            abs1 = np.abs(total1)
            abs2 = np.abs(total2)
            
            # Set invalid to infinity
            abs1[~valid1] = np.iinfo(np.int64).max
            abs2[~valid2] = np.iinfo(np.int64).max
            
            # Find global min index
            # We need to check both arrays.
            # Create a combined view?
            # Or just find min of min(abs1, abs2)
            
            min_abs1 = np.min(abs1)
            min_abs2 = np.min(abs2)
            
            if min_abs1 < min_abs2:
                best_val = min_abs1
                idx_min = np.argmin(abs1)
                best_s1 = sums1[idx_min]
                best_s2 = cand1_s2[idx_min]
            else:
                best_val = min_abs2
                idx_min = np.argmin(abs2)
                best_s1 = sums1[idx_min]
                best_s2 = cand2_s2[idx_min]
                
    except Exception:
        # Fallback to simple heuristic if something goes wrong or is too slow
        return [0]*28 + [1, -1]
    
    if time.time() - start_time > timeout * 0.7:
        # Return fallback if taking too long
        return [0]*28 + [1, -1]
        
    def find_coeffs(w, target):
        n = len(w)
        coeffs = [0] * n
        # Precompute suffix sums of absolute values for pruning
        suffix_abs = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_abs[i] = suffix_abs[i + 1] + abs(w[i])
        
        def dfs(pos, current):
            if pos == n:
                return current == target
            
            # Pruning: Check if target is reachable with remaining weights
            if abs(target - current) > suffix_abs[pos]:
                return False
            
            wi = w[pos]
            # Try -1, 0, 1
            # Ordering can matter for speed; trying 0 first often finds sparse solutions
            # But here we want any solution.
            for c in (-1, 0, 1):
                coeffs[pos] = c
                if dfs(pos + 1, current + c * wi):
                    return True
            coeffs[pos] = 0
            return False
            
        if dfs(0, 0):
            return coeffs
        return None
        
    c1 = find_coeffs(w1, best_s1)
    c2 = find_coeffs(w2, best_s2)
    
    if c1 is None or c2 is None:
        return [0]*28 + [1, -1]
        
    return c1 + c2

# EVOLVE-BLOCK-END
