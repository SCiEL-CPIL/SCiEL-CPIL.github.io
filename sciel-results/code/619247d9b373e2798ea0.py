import time
import numpy as np
from math import gcd

def _lcm_range(n):
    result = 1
    for i in range(2, n + 1):
        result = result * i // gcd(result, i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    n = 30
    lcm_val = _lcm_range(n)
    
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    half = 15
    w1 = np.array(weights[:half], dtype=np.int64)
    w2 = np.array(weights[half:], dtype=np.int64)
    
    def gen_all_sums(w_part):
        n_part = len(w_part)
        sums = np.array([0], dtype=np.int64)
        for i in range(n_part):
            a_i = w_part[i]
            sums = np.concatenate([sums + a_i, sums, sums - a_i])
        return sums

    if time.time() > deadline - 15:
        return [0] * 28 + [1, -1]

    sums1 = gen_all_sums(w1)
    sums1.sort()
    
    if time.time() > deadline - 15:
        return [0] * 28 + [1, -1]

    sums2 = gen_all_sums(w2)
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    n1 = len(sums1)
    n2 = len(sums2)
    
    chunk_size = 100000
    
    for chunk_start in range(0, n2, chunk_size):
        if time.time() > deadline - 5:
            break
        
        chunk_end = min(chunk_start + chunk_size, n2)
        s2_chunk = sums2[chunk_start:chunk_end]
        targets = -s2_chunk
        
        pos = np.searchsorted(sums1, targets)
        
        # Candidates: pos-1, pos, pos+1
        # Clip to valid range [0, n1-1]
        
        # We will create arrays for the 3 candidates
        # To avoid huge memory spikes, we can process in smaller sub-chunks or just let NumPy handle it
        # 100k targets * 3 candidates = 300k elements. Fine.
        
        pos_c1 = np.clip(pos - 1, 0, n1 - 1)
        pos_c2 = np.clip(pos, 0, n1 - 1)
        pos_c3 = np.clip(pos + 1, 0, n1 - 1)
        
        s1_c1 = sums1[pos_c1]
        s1_c2 = sums1[pos_c2]
        s1_c3 = sums1[pos_c3]
        
        total1 = s1_c1 + s2_chunk
        total2 = s1_c2 + s2_chunk
        total3 = s1_c3 + s2_chunk
        
        abs1 = np.abs(total1)
        abs2 = np.abs(total2)
        abs3 = np.abs(total3)
        
        # Stack them to find global min
        # Shape: (3, chunk_size)
        all_abs = np.vstack([abs1, abs2, abs3])
        all_s1 = np.vstack([s1_c1, s1_c2, s1_c3])
        all_s2 = np.vstack([s2_chunk, s2_chunk, s2_chunk])
        all_total = np.vstack([total1, total2, total3])
        
        # Mask out zeros
        nonzero_mask = all_total != 0
        if not np.any(nonzero_mask):
            continue
            
        abs_nonzero = np.where(nonzero_mask, all_abs, np.iinfo(np.int64).max)
        
        min_val = np.min(abs_nonzero)
        
        if best_abs is None or min_val < best_abs:
            best_abs = min_val
            # Find indices of the minimum
            # Note: min_val might appear multiple times
            min_indices = np.where(abs_nonzero == min_val)
            # Pick the first one
            idx = min_indices[0][0]
            row = min_indices[1][0]
            col = min_indices[0][0] # Wait, np.where on 2D array returns (row_indices, col_indices)
            # Actually, np.where(condition) returns tuple of arrays.
            # Let's just use argmin on the flattened array or find the location properly.
            
            # Simpler:
            flat_abs = abs_nonzero.flatten()
            min_idx_flat = np.argmin(flat_abs)
            row_idx = min_idx_flat // len(s2_chunk)
            col_idx = min_idx_flat % len(s2_chunk)
            
            best_s1 = int(all_s1[row_idx, col_idx])
            best_s2 = int(all_s2[row_idx, col_idx])
            
            if best_abs == 1:
                break

    if best_s1 is None:
        return [0] * 28 + [1, -1]
    
    def find_coeffs(target, w_part):
        n_part = len(w_part)
        coeffs = [0] * n_part
        
        def dfs(idx, current_sum):
            if idx == n_part:
                return current_sum == target
            
            a_i = w_part[idx]
            rem = target - current_sum
            
            # Heuristic: try values that get us closer to target
            if rem > 0:
                order = [1, 0, -1]
            elif rem < 0:
                order = [-1, 0, 1]
            else:
                order = [0, 1, -1]
            
            for val in order:
                coeffs[idx] = val
                if dfs(idx + 1, current_sum + val * a_i):
                    return True
            
            return False
        
        if dfs(0, 0):
            return coeffs
        return None
    
    c1 = find_coeffs(best_s1, w1)
    c2 = find_coeffs(best_s2, w2)
    
    if c1 is None or c2 is None:
        return [0] * 28 + [1, -1]
    
    result = [0] * 30
    for i in range(half):
        result[i] = int(c1[i])
    for i in range(half):
        result[half + i] = int(c2[i])
    
    return result
