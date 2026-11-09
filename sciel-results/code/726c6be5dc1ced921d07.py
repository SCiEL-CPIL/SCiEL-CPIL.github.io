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
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 30
    lcm_val = _lcm_range(n)
    
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    # c_i in {-1, 0, 1}
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Split into two halves for meet-in-the-middle
    # First half: indices 0..14 (denominators 1..15)
    # Second half: indices 15..29 (denominators 16..30)
    half = 15
    w1 = np.array(weights[:half], dtype=np.int64)
    w2 = np.array(weights[half:], dtype=np.int64)
    
    # Generate all possible sums for the first half
    # There are 3^15 = 14,348,907 combinations.
    def gen_all_sums(w_part):
        n_part = len(w_part)
        sums = np.array([0], dtype=np.int64)
        for i in range(n_part):
            a_i = w_part[i]
            # Efficiently create new array by concatenating
            # sums + a_i, sums, sums - a_i
            sums = np.concatenate([sums + a_i, sums, sums - a_i])
        return sums

    # Check time
    if time.time() > deadline - 10:
        # Fallback if generation takes too long (unlikely)
        return [0] * 28 + [1, -1]

    sums1 = gen_all_sums(w1)
    sums1.sort()
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]

    sums2 = gen_all_sums(w2)
    
    # For each s2, find s1 such that s1 + s2 is closest to 0 but not 0
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    n1 = len(sums1)
    n2 = len(sums2)
    
    # Process in chunks to manage memory and allow time checks
    chunk_size = 50000
    
    for chunk_start in range(0, n2, chunk_size):
        if time.time() > deadline - 5:
            break
        
        chunk_end = min(chunk_start + chunk_size, n2)
        s2_chunk = sums2[chunk_start:chunk_end]
        targets = -s2_chunk
        
        # Find insertion points in sorted sums1
        pos = np.searchsorted(sums1, targets)
        
        # Candidate 1: sums1[pos] (clipped)
        pos_clipped = np.clip(pos, 0, n1 - 1)
        s1_cand1 = sums1[pos_clipped]
        total1 = s1_cand1 + s2_chunk
        abs_total1 = np.abs(total1)
        
        # Candidate 2: sums1[pos-1] (clipped)
        pos_minus = np.clip(pos - 1, 0, n1 - 1)
        s1_cand2 = sums1[pos_minus]
        total2 = s1_cand2 + s2_chunk
        abs_total2 = np.abs(total2)
        
        # Combine and find minimum nonzero
        # We want to minimize abs(total) where total != 0
        all_abs = np.concatenate([abs_total1, abs_total2])
        all_s1 = np.concatenate([s1_cand1, s1_cand2])
        all_s2 = np.concatenate([s2_chunk, s2_chunk])
        all_total = np.concatenate([total1, total2])
        
        # Mask out zeros
        nonzero_mask = all_total != 0
        if not np.any(nonzero_mask):
            continue
        
        # Set zeros to infinity so they are not selected
        abs_nonzero = np.where(nonzero_mask, all_abs, np.iinfo(np.int64).max)
        min_idx = np.argmin(abs_nonzero)
        min_abs = int(abs_nonzero[min_idx])
        
        if best_abs is None or min_abs < best_abs:
            best_abs = min_abs
            best_s1 = int(all_s1[min_idx])
            best_s2 = int(all_s2[min_idx])
            
            if best_abs == 1:
                # Can't do better than 1 (in integer units)
                break
    
    if best_s1 is None:
        # Fallback: simple nonzero sum
        return [0] * 28 + [1, -1]
    
    # Reconstruct coefficients
    def find_coeffs(target, w_part):
        n_part = len(w_part)
        coeffs = [0] * n_part
        
        def dfs(idx, current_sum):
            if idx == n_part:
                return current_sum == target
            
            a_i = w_part[idx]
            rem = target - current_sum
            
            # Try values in order of proximity to remaining target
            # This heuristic helps find a solution faster
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
