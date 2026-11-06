"""Program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 256
    signs = [0] * n
    D_old = 0.0
    
    # prefix_sums[d][i] stores the sum of the subsequence starting at 1 (or 0) 
    # with step d, up to index i (1-indexed).
    # Actually, let's use 1-indexed for clarity in logic, but 0-indexed for array.
    # prefix_sums[d][i] = sum of s_j for j = r, r+d, ..., i where i % d == r % d?
    # No, simpler: prefix_sums[d][i] = sum of s_{i-kd} for k=0,1,... such that index >= 1.
    # This is the sum of the AP ending at i with step d and starting as far left as possible.
    # Wait, that's not enough. We need sum from arbitrary start a.
    # Sum(a, i) = prefix_sums[d][i] - prefix_sums[d][a-d].
    
    # We need to store prefix_sums for all d and i.
    # d from 1 to n-1. i from 1 to n.
    # Size: 256 * 256 = 65536 floats. Small.
    
    # Initialize prefix_sums
    # prefix_sums[d][i] for i=1..n
    # We will build it incrementally.
    
    # For each d, we have d residues.
    # Let's just store a 2D array ps[d][i]
    ps = [[0.0] * (n + 1) for _ in range(n + 1)]
    
    for i in range(n):
        pos = i + 1
        best_val = float('inf')
        best_sign = 1
        
        for s_i in (1, -1):
            M_s = 0.0
            
            # Update ps for this s_i? 
            # No, we need to evaluate M_s based on hypothetical s_i.
            # We can calculate the new ps values on the fly for evaluation.
            
            for d in range(1, pos):
                # For step d, the APs ending at pos start at a = pos, pos-d, ...
                # Sum(a, pos) = ps_new[d][pos] - ps[d][a-d]
                # ps_new[d][pos] = ps[d][pos-d] + s_i  (if pos-d >= 1) else s_i
                # Note: ps[d][x] is the sum of the subsequence ending at x with step d.
                
                # Calculate ps_new_pos
                if pos - d >= 1:
                    ps_new_pos = ps[d][pos - d] + s_i
                else:
                    ps_new_pos = s_i
                
                # Now check all starts a
                # a = pos, pos-d, pos-2d, ... >= 1
                # Sum(a, pos) = ps_new_pos - ps[d][a-d]
                # If a=1, a-d <= 0, ps[d][a-d] = 0.
                
                a = pos
                while a >= 1:
                    if a - d >= 1:
                        prev_sum = ps[d][a - d]
                    else:
                        prev_sum = 0.0
                    
                    current_sum = ps_new_pos - prev_sum
                    abs_sum = abs(current_sum)
                    if abs_sum > M_s:
                        M_s = abs_sum
                    
                    a -= d
            
            D_new = max(D_old, M_s)
            if D_new < best_val:
                best_val = D_new
                best_sign = s_i
            elif D_new == best_val:
                if random.random() < 0.5:
                    best_sign = s_i
        
        signs[i] = best_sign
        
        # Now update ps with the chosen best_sign
        for d in range(1, pos):
            if pos - d >= 1:
                ps[d][pos] = ps[d][pos - d] + best_sign
            else:
                ps[d][pos] = best_sign
        
        D_old = best_val

    result = signs
    assert len(result) == n
    assert all(s in (-1, 1) for s in result)
    return result
# EVOLVE-BLOCK-END
