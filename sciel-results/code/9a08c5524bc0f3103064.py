import time
import math
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    # 90s limit, keep 5s safety margin
    deadline = start_time + 85.0
    
    n = 30
    
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Weights w_i = L / (i+1)
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    n1 = n // 2
    n2 = n - n1
    
    # Helper to generate sums and codes
    def generate_half(n_vars, start_idx):
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i in range(n_vars):
            w = weights[start_idx + i]
            # Expand
            # New sums: s, s+w, s-w
            # New codes: c, c+1<<(2*i), c+2<<(2*i)
            
            len_old = len(sums)
            s_new = np.empty(len_old * 3, dtype=np.int64)
            c_new = np.empty(len_old * 3, dtype=np.int64)
            
            s_new[0::3] = sums
            s_new[1::3] = sums + w
            s_new[2::3] = sums - w
            
            c_new[0::3] = codes
            c_new[1::3] = codes + (1 << (2 * i))
            c_new[2::3] = codes + (2 << (2 * i))
            
            sums = s_new
            codes = c_new
            
        return sums, codes

    # Generate first half
    sums1, codes1 = generate_half(n1, 0)
    
    # Generate second half
    sums2, codes2 = generate_half(n2, n1)
    
    # Sort second half by sum
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    codes2_sorted = codes2[sort_idx]
    
    # Search for best pair
    # Target for each s1 is -s1
    targets = -sums1
    
    # Find insertion points in sorted sums2
    positions = np.searchsorted(sums2_sorted, targets)
    
    # Check left and right neighbors
    # pos_left: positions - 1
    # pos_right: positions
    # Clip to valid range [0, len-1]
    
    pos_left = np.clip(positions - 1, 0, len(sums2_sorted) - 1)
    pos_right = np.clip(positions, 0, len(sums2_sorted) - 1)
    
    # Get candidate sums
    s2_left = sums2_sorted[pos_left]
    s2_right = sums2_sorted[pos_right]
    
    # Get candidate codes
    c2_left = codes2_sorted[pos_left]
    c2_right = codes2_sorted[pos_right]
    
    # Calculate total sums
    total_left = sums1 + s2_left
    total_right = sums1 + s2_right
    
    # Calculate absolute sums
    abs_left = np.abs(total_left)
    abs_right = np.abs(total_right)
    
    # Mask out zero sums
    mask_left = total_left != 0
    mask_right = total_right != 0
    
    # Set invalid ones to max
    max_int = np.iinfo(np.int64).max
    abs_left[~mask_left] = max_int
    abs_right[~mask_right] = max_int
    
    # Find global minimum
    min_abs_left = np.min(abs_left)
    min_abs_right = np.min(abs_right)
    
    best_val = max_int
    best_c1 = 0
    best_c2 = 0
    
    if min_abs_left < min_abs_right:
        best_idx = np.argmin(abs_left)
        best_val = min_abs_left
        best_c1 = codes1[best_idx]
        best_c2 = c2_left[best_idx]
    else:
        best_idx = np.argmin(abs_right)
        best_val = min_abs_right
        best_c1 = codes1[best_idx]
        best_c2 = c2_right[best_idx]
        
    # Reconstruct coefficients
    coeffs = [0] * n
    
    # Decode code1
    c = best_c1
    for i in range(n1):
        val = (c >> (2 * i)) & 3
        if val == 1:
            coeffs[i] = 1
        elif val == 2:
            coeffs[i] = -1
            
    # Decode code2
    c = best_c2
    for i in range(n2):
        val = (c >> (2 * i)) & 3
        if val == 1:
            coeffs[n1 + i] = 1
        elif val == 2:
            coeffs[n1 + i] = -1
            
    # Optional: Verify and Polish (if time permits)
    # Calculate exact sum to ensure non-zero
    exact_sum = Fraction(0, 1)
    for i in range(n):
        if coeffs[i] != 0:
            exact_sum += Fraction(coeffs[i], i + 1)
            
    if exact_sum == 0:
        # Fallback if somehow zero (should not happen with MITM mask)
        return [0] * 28 + [1, -1]
        
    return coeffs
