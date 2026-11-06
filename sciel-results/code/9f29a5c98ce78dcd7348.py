"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import gcd

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    N = 30
    half = 15
    
    # Denominators are 1, 2, ..., 30
    # Use global LCM as common denominator
    # LCM(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    # L = 465817912560
    
    # For each denominator d, the coefficient c/d = c * (L/d) / L
    # So we need integer weights w[i] = L // (i+1) for i=0..29
    weights = np.array([L // (i + 1) for i in range(N)], dtype=np.int64)
    
    # Split into two halves
    w1 = weights[:half]  # denominators 1..15
    w2 = weights[half:]  # denominators 16..30
    
    def gen_sums(weights_arr, time_limit):
        """Generate all possible sums for given weights with coefficients in {-1,0,1}."""
        n = len(weights_arr)
        # Use iterative approach to generate all 3^n combinations
        # Represent as base-3 digits: 0 -> -1, 1 -> 0, 2 -> +1
        
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        
        for w in weights_arr:
            # For each existing sum, create three new sums: sum-w, sum, sum+w
            # Vectorized: new_sums = concat(sums - w, sums, sums + w)
            sums = np.concatenate([sums - w, sums, sums + w])
            
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        return sums
    
    # Generate sums for half 1
    sums1 = gen_sums(w1, deadline - 10.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Fallback: return a simple valid solution
        return [0]*28 + [1, -1]
    
    # Generate sums for half 2
    sums2 = gen_sums(w2, deadline - 5.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 2.0:
        return [0]*28 + [1, -1]
    
    # Sort sums1 for binary search
    sums1_sorted = np.sort(sums1)
    
    # For each value in sums2, find the value in sums1 that makes sum closest to 0
    # We want a + b ≈ 0, i.e., a ≈ -b
    # For each b in sums2, find a in sums1_sorted closest to -b
    
    # Use searchsorted to find insertion point
    neg_sums2 = -sums2
    # Find where each -b would be inserted in sums1_sorted
    idx = np.searchsorted(sums1_sorted, neg_sums2)
    
    # Check candidates: idx-1 and idx (and possibly idx+1 for safety)
    best_abs = None
    best_a = None
    best_b = None
    best_idx_a = None
    best_idx_b = None
    
    # Process in chunks to manage memory
    chunk_size = 100000
    for start in range(0, len(sums2), chunk_size):
        end = min(start + chunk_size, len(sums2))
        neg_chunk = neg_sums2[start:end]
        idx_chunk = idx[start:end]
        
        # Candidate indices: idx-1 and idx
        # Clamp to valid range
        idx_left = np.clip(idx_chunk - 1, 0, len(sums1_sorted) - 1)
        idx_right = np.clip(idx_chunk, 0, len(sums1_sorted) - 1)
        
        a_left = sums1_sorted[idx_left]
        a_right = sums1_sorted[idx_right]
        
        b_chunk = sums2[start:end]
        
        # Compute sums
        sum_left = a_left + b_chunk
        sum_right = a_right + b_chunk
        
        abs_left = np.abs(sum_left)
        abs_right = np.abs(sum_right)
        
        # Exclude zero sums
        abs_left[sum_left == 0] = np.iinfo(np.int64).max
        abs_right[sum_right == 0] = np.iinfo(np.int64).max
        
        # Find best in this chunk
        min_left = np.min(abs_left)
        min_right = np.min(abs_right)
        
        if min_left < min_right:
            local_best_idx = np.argmin(abs_left)
            local_best_abs = min_left
            local_best_a = a_left[local_best_idx]
            local_best_b = b_chunk[local_best_idx]
            local_best_idx_a = idx_left[local_best_idx]
            local_best_idx_b = start + local_best_idx
        else:
            local_best_idx = np.argmin(abs_right)
            local_best_abs = min_right
            local_best_a = a_right[local_best_idx]
            local_best_b = b_chunk[local_best_idx]
            local_best_idx_a = idx_right[local_best_idx]
            local_best_idx_b = start + local_best_idx
        
        if best_abs is None or local_best_abs < best_abs:
            best_abs = local_best_abs
            best_a = local_best_a
            best_b = local_best_b
            best_idx_a = local_best_idx_a
            best_idx_b = local_best_idx_b
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
    
    if best_abs is None or best_abs == np.iinfo(np.int64).max:
        # No valid solution found, fallback
        return [0]*28 + [1, -1]
    
    # Now reconstruct the coefficients
    # best_a is a sum from half 1, best_b from half 2
    # We need to find which combination of coefficients gives best_a
    
    # For half 1: find coefficients that sum to best_a
    # This requires searching through the 3^15 combinations
    # We can do this by backtracking
    
    def find_coeffs(weights_arr, target):
        """Find coefficients in {-1,0,1} that sum to target using weights_arr."""
        n = len(weights_arr)
        # Recursive search with pruning
        # Use iterative approach for speed
        
        # Generate all combinations and find the one matching target
        # For n=15, 3^15 = 14.3M, we can enumerate
        
        # Actually, let's use a smarter approach:
        # Process weights one by one, maintaining possible partial sums
        # But for exact match, we can just enumerate
        
        # Use numpy to generate all base-3 representations
        # For n=15, this is 14.3M entries
        
        # Alternative: use the fact that we already have sums1_sorted
        # We need to find WHICH combination gives best_a
        
        # Let's enumerate using a different method
        # Convert target to find the right combination
        
        # Simple approach: iterate through all 3^n combinations
        # Represent combination as base-3 number
        # For n=15, max base-3 number is 3^15 - 1 = 14348906
        
        result = [0] * n
        remaining = target
        
        # Backtracking from last weight to first
        for i in range(n - 1, -1, -1):
            w = weights_arr[i]
            for c in [1, 0, -1]:
                new_remaining = remaining - c * w
                # Check if new_remaining is achievable with remaining weights
                # Max possible from weights[0:i] is sum(abs(weights[0:i]))
                if i > 0:
                    max_possible = np.sum(np.abs(weights_arr[:i]))
                    if abs(new_remaining) <= max_possible:
                        result[i] = c
                        remaining = new_remaining
                        break
            else:
                # No valid coefficient found, try all
                for c in [1, 0, -1]:
                    result[i] = c
                    remaining = target - sum(result[j] * weights_arr[j] for j in range(i+1))
                    # This is getting complicated, let's just enumerate
        
        # Let's use a cleaner enumeration approach
        # For 15 elements, enumerate all 3^15 = 14.3M combinations
        # Use vectorized numpy
        
        # Actually, let me use a simpler recursive approach with memoization
        # Or just do a direct search
        
        # Direct search: try all combinations
        # For n=15, this is feasible
        coeffs = [0] * n
        found = False
        
        def backtrack(pos, current_sum):
            nonlocal found, coeffs
            if found:
                return
            if pos == n:
                if current_sum == target:
                    found = True
                    return
                return
            
            # Pruning: check if remaining can achieve target
            remaining_weights = weights_arr[pos:]
            max_rem = np.sum(np.abs(remaining_weights))
            needed = target - current_sum
            if abs(needed) > max_rem:
                return
            
            for c in [1, 0, -1]:
                coeffs[pos] = c
                backtrack(pos + 1, current_sum + c * weights_arr[pos])
                if found:
                    return
        
        backtrack(0, 0)
        return coeffs
    
    # Find coefficients for half 1
    coeffs1 = find_coeffs(w1, best_a)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 3.0:
        return [0]*28 + [1, -1]
    
    # Find coefficients for half 2
    coeffs2 = find_coeffs(w2, best_b)
    
    # Combine
    result = list(coeffs1) + list(coeffs2)
    
    # Verify
    total = sum(result[i] / (i + 1) for i in range(N))
    if total == 0:
        # Shouldn't happen, but fallback
        return [0]*28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
