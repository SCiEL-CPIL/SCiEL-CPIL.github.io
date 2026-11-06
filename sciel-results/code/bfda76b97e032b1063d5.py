import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    H = 1
    for i in range(1, n + 1):
        H = H * i // math.gcd(H, i)
        
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = n // 2  # 15
    
    def generate_half(start, end):
        """Generates list of (sum, mask) for indices start to end-1."""
        length = end - start
        # Pre-compute powers of 3
        powers3 = [1] * length
        for i in range(1, length):
            powers3[i] = powers3[i-1] * 3
        
        sums = [0]
        masks = [0]
        
        for i in range(length):
            w = W[start + i]
            new_sums = []
            new_masks = []
            mult = powers3[i]
            for k in range(len(sums)):
                s = sums[k]
                m = masks[k]
                
                # c = -1 (digit 0)
                new_sums.append(s - w)
                new_masks.append(m)
                
                # c = 0 (digit 1)
                new_sums.append(s)
                new_masks.append(m + mult)
                
                # c = 1 (digit 2)
                new_sums.append(s + w)
                new_masks.append(m + 2 * mult)
                
            sums = new_sums
            masks = new_masks
        return sums, masks

    # Generate Half A (indices 0 to 14)
    sums_a, masks_a = generate_half(0, mid)
    
    if time.time() - start_time > deadline:
        # Fallback if generation took too long
        coeffs = [0] * n
        coeffs[0] = 1
        return coeffs
    
    # Generate Half B (indices 15 to 29)
    sums_b, masks_b = generate_half(mid, n)
    
    if time.time() - start_time > deadline:
        # Fallback
        coeffs = [0] * n
        coeffs[0] = 1
        return coeffs
    
    # Sort Half B by sum
    pairs_b = list(zip(sums_b, masks_b))
    pairs_b.sort(key=lambda x: x[0])
    
    arr_b_sums = np.array([p[0] for p in pairs_b])
    arr_b_masks = [p[1] for p in pairs_b]
    
    best_val = float('inf')
    best_mask_a = -1
    best_mask_b = -1
    
    # Use numpy searchsorted for efficient nearest-neighbor search
    for i in range(len(sums_a)):
        if time.time() - start_time > deadline:
            break
            
        a_sum = sums_a[i]
        target = -a_sum
        
        # Find insertion point
        idx = np.searchsorted(arr_b_sums, target)
        
        # Check neighbors: idx-2, idx-1, idx, idx+1, idx+2
        candidates = set()
        for offset in [-2, -1, 0, 1, 2]:
            c_idx = idx + offset
            if 0 <= c_idx < len(arr_b_sums):
                candidates.add(c_idx)
        
        for c_idx in candidates:
            b_sum = arr_b_sums[c_idx]
            total = a_sum + b_sum
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_val:
                    best_val = abs_total
                    best_mask_a = masks_a[i]
                    best_mask_b = arr_b_masks[c_idx]
    
    # Reconstruct coefficients
    coeffs = [0] * n
    
    if best_mask_a != -1:
        # Decode mask_a
        for i in range(mid):
            digit = (best_mask_a // (3 ** i)) % 3
            coeffs[i] = digit - 1
            
        # Decode mask_b
        for i in range(mid):
            digit = (best_mask_b // (3 ** i)) % 3
            coeffs[mid + i] = digit - 1
        
        # Verify the sum
        s = sum(coeffs[i] * W[i] for i in range(n))
        if s == 0:
            # Should not happen, but fallback
            coeffs[0] = 1
            coeffs[1] = -1
    else:
        # Fallback: simple non-zero solution
        coeffs[0] = 1
        coeffs[1] = -1
        s = sum(coeffs[i] * W[i] for i in range(n))
        if s == 0:
            coeffs[2] = 1

    return coeffs
