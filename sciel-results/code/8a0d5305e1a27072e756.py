import time
import math
import numpy as np

def construct():
        start_time = time.time()
        deadline = start_time + 85.0
        
        n = 30
        L = 1
        for i in range(1, n + 1):
            L = L * i // math.gcd(L, i)
        
        weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
        
        n1 = n // 2
        n2 = n - n1
        
        # Generate sums for first half
        # Using a list of arrays might be slower, but let's stick to the vectorized expansion
        sums1 = np.array([0], dtype=np.int64)
        codes1 = np.array([0], dtype=np.int64)
        
        for i in range(n1):
            w = weights[i]
            len_curr = len(sums1)
            
            # Preallocate
            sums1_new = np.empty(len_curr * 3, dtype=np.int64)
            codes1_new = np.empty(len_curr * 3, dtype=np.int64)
            
            sums1_new[0::3] = sums1
            sums1_new[1::3] = sums1 + w
            sums1_new[2::3] = sums1 - w
            
            codes1_new[0::3] = codes1
            codes1_new[1::3] = codes1 + (1 << (2 * i))
            codes1_new[2::3] = codes1 + (2 << (2 * i))
            
            sums1 = sums1_new
            codes1 = codes1_new
            
            if time.time() > deadline - 30:
                break # Safety break, though unlikely to trigger here
        
        # Generate sums for second half
        sums2 = np.array([0], dtype=np.int64)
        codes2 = np.array([0], dtype=np.int64)
        
        for i in range(n2):
            w = weights[n1 + i]
            len_curr = len(sums2)
            
            sums2_new = np.empty(len_curr * 3, dtype=np.int64)
            codes2_new = np.empty(len_curr * 3, dtype=np.int64)
            
            sums2_new[0::3] = sums2
            sums2_new[1::3] = sums2 + w
            sums2_new[2::3] = sums2 - w
            
            codes2_new[0::3] = codes2
            codes2_new[1::3] = codes2 + (1 << (2 * i))
            codes2_new[2::3] = codes2 + (2 << (2 * i))
            
            sums2 = sums2_new
            codes2 = codes2_new
            
            if time.time() > deadline - 20:
                break
        
        # Sort second half
        sort_idx = np.argsort(sums2)
        sums2_sorted = sums2[sort_idx]
        codes2_sorted = codes2[sort_idx]
        
        # Search
        targets = -sums1
        positions = np.searchsorted(sums2_sorted, targets)
        
        pos_left = np.clip(positions - 1, 0, len(sums2_sorted) - 1)
        pos_right = np.clip(positions, 0, len(sums2_sorted) - 1)
        
        s2_left = sums2_sorted[pos_left]
        s2_right = sums2_sorted[pos_right]
        c2_left = codes2_sorted[pos_left]
        c2_right = codes2_sorted[pos_right]
        
        total_left = sums1 + s2_left
        total_right = sums1 + s2_right
        
        abs_left = np.abs(total_left)
        abs_right = np.abs(total_right)
        
        mask_left = total_left != 0
        mask_right = total_right != 0
        
        abs_left[~mask_left] = np.iinfo(np.int64).max
        abs_right[~mask_right] = np.iinfo(np.int64).max
        
        min_abs_left = np.min(abs_left)
        min_abs_right = np.min(abs_right)
        
        if min_abs_left < min_abs_right:
            best_idx = np.argmin(abs_left)
            best_val = min_abs_left
            s2_best = s2_left[best_idx]
            c2_best = c2_left[best_idx]
            c1_best = codes1[best_idx]
        else:
            best_idx = np.argmin(abs_right)
            best_val = min_abs_right
            s2_best = s2_right[best_idx]
            c2_best = c2_right[best_idx]
            c1_best = codes1[best_idx]
            
        # Reconstruct
        coeffs = [0] * n
        c = c1_best
        for i in range(n1):
            val = (c >> (2 * i)) & 3
            if val == 1: coeffs[i] = 1
            elif val == 2: coeffs[i] = -1
            
        c = c2_best
        for i in range(n2):
            val = (c >> (2 * i)) & 3
            if val == 1: coeffs[n1 + i] = 1
            elif val == 2: coeffs[n1 + i] = -1
            
        # Verify
        s = sum(coeffs[i] / (i + 1) for i in range(n))
        if s == 0:
             return [0]*28 + [1, -1]
             
        return coeffs
