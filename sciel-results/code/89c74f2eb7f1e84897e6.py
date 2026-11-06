import math
import numpy as np
import time
from functools import reduce

def construct():
    start_time = time.time()
    n = 30
    
    # Calculate LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Suffix sums R[i] = sum(W[i-1] ... W[n-1])
    # R[n+1] = 0
    R = [0] * (n + 2)
    for i in range(n, 0, -1):
        R[i] = R[i + 1] + W[i - 1]

    # Split into two halves: 1-15 and 16-30
    mid = 15
    
    # Generate all possible sums for first half (1-15)
    # We use a set to store unique sums
    sums_a = {0}
    for i in range(1, mid + 1):
        w = W[i - 1]
        new_sums = set()
        for s in sums_a:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_a = new_sums
        
    # Generate all possible sums for second half (16-30)
    sums_b = {0}
    for i in range(mid + 1, n + 1):
        w = W[i - 1]
        new_sums = set()
        for s in sums_b:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_b = new_sums
        
    # To speed up lookup, ensure sums_b is a set (it is).
    # We will iterate over sums_a and check if (target - a) is in sums_b.
    
    # Greedy initialization for fallback
    curr_sum = 0
    coeffs_greedy = [0] * n
    for i in range(n, 0, -1):
        w = W[i - 1]
        best_c = 0
        best_val_local = curr_sum
        for c in [-1, 0, 1]:
            val = curr_sum + c * w
            if abs(val) < abs(best_val_local):
                best_val_local = val
                best_c = c
        coeffs_greedy[i - 1] = best_c
        curr_sum = best_val_local
        
    if curr_sum != 0:
        best_val = abs(curr_sum)
    else:
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs_greedy[min_idx] = 1
        curr_sum = sum(c * W[i] for i, c in enumerate(coeffs_greedy))
        best_val = abs(curr_sum)
        
    best_coeffs = coeffs_greedy.copy()
    
    # Search for small integer sums k
    # The sum S = sum(c_i * W_i) = H * sum(c_i/i).
    # We want to minimize |S|/H, which is equivalent to minimizing |S|.
    # S is an integer. Smallest non-zero |S| is 1.
    
    # We check k = 1, 2, ...
    # For a given k, we need a in sums_a such that k-a in sums_b or -k-a in sums_b.
    
    max_k = 1000 # Limit k to avoid infinite loop if 1 is not possible (unlikely)
    
    for k in range(1, max_k + 1):
        # Check time
        if time.time() - start_time > 50:
            break
            
        target_pos = k
        target_neg = -k
        
        # Iterate over sums_a
        # To make it faster, we can pick the smaller set to iterate?
        # sums_a and sums_b are similar size.
        # We check if (target - a) in sums_b
        
        found_a = None
        found_b = None
        found_sign = 1 # 1 for +k, -1 for -k
        
        # Optimization: If sums_a is large, iterating it is slow.
        # But we have to do it.
        for a in sums_a:
            # Check positive target
            b_pos = target_pos - a
            if b_pos in sums_b:
                found_a = a
                found_b = b_pos
                found_sign = 1
                break
                
            # Check negative target
            b_neg = target_neg - a
            if b_neg in sums_b:
                found_a = a
                found_b = b_neg
                found_sign = -1
                break
                
            # Time check inside loop
            if time.time() - start_time > 50:
                break
                
        if found_a is not None:
            # Reconstruct coefficients
            coeffs_a = find_coeffs(found_a, 1, mid, W, R)
            coeffs_b = find_coeffs(found_b, mid + 1, n, W, R)
            
            if coeffs_a is not None and coeffs_b is not None:
                final_coeffs = coeffs_a + coeffs_b
                # Verify
                total = sum(c * W[i] for i, c in enumerate(final_coeffs))
                if total == found_sign * k:
                    return final_coeffs
                # If verification fails (shouldn't happen), continue or fallback
            else:
                # Reconstruction failed? Should not happen if a in sums_a.
                # Maybe find_coeffs has a bug or timeout?
                pass
        else:
            # If we broke out of 'for a' due to time, break outer loop too
            if time.time() - start_time > 50:
                break

    # Fallback to greedy
    # Ensure greedy is non-zero
    total_greedy = sum(c * W[i] for i, c in enumerate(best_coeffs))
    if total_greedy == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if best_coeffs[min_idx] == 1:
            best_coeffs[min_idx] = -1
        else:
            best_coeffs[min_idx] = 1
            
    return best_coeffs

def find_coeffs(target, start, end, W, R):
    """
    Recursively find coefficients for indices start to end (1-based)
    such that sum(c_i * W[i-1]) == target.
    """
    if start > end:
        return [] if target == 0 else None
        
    # Pruning: max possible sum from start to end
    # R[i] is sum of W[i-1] to W[n-1]
    # Sum of W[start-1] to W[end-1] is R[start] - R[end+1]
    max_rem = R[start] - R[end + 1]
    if abs(target) > max_rem:
        return None
        
    w = W[start - 1]
    for c in [-1, 0, 1]:
        res = find_coeffs(target - c * w, start + 1, end, W, R)
        if res is not None:
            return [c] + res
    return None
