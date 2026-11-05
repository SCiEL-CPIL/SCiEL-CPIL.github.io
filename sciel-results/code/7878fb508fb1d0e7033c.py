import math
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    LIMIT = 145.0  # seconds (safety margin for 150s limit)
    MAX_S = 10**12
    MAX_K = 200
    
    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Compute phi for a single number using trial division with primes
    def compute_phi_single(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result
    
    # 3. Compute phi for a segment using vectorized approach
    def compute_phi_segment(L, R):
        size = R - L + 1
        phi_vals = np.arange(L, R + 1, dtype=np.int64)
        rem = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            indices = np.arange(start, R + 1, p) - L
            phi_vals[indices] = phi_vals[indices] // p * (p - 1)
            mask = indices.copy()
            while True:
                vals = rem[mask]
                divisible = vals % p == 0
                if not np.any(divisible):
                    break
                new_mask = mask[divisible]
                rem[new_mask] = rem[new_mask] // p
                mask = new_mask
        
        remaining = rem > 1
        if np.any(remaining):
            idxs = np.where(remaining)[0]
            phi_vals[idxs] = phi_vals[idxs] // rem[idxs] * (rem[idxs] - 1)
                
        return phi_vals.tolist()
    
    # 4. Find longest run of distinct values in a list
    def find_longest_run(phis, max_len_check=200):
        seen_vals = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        
        for r in range(len(phis)):
            v = phis[r]
            if v in seen_vals:
                prev_idx = seen_vals[v]
                if prev_idx >= l:
                    l = prev_idx + 1
            seen_vals[v] = r
            
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start_idx = l
            
            if max_len >= max_len_check:
                break
        
        return max_len, max_start_idx
    
    # 5. Search strategy: primorial-based arithmetic progression + random sampling
    best_k = 1
    best_s = 1
    
    # Primorial: 2*3*5*7*11*13*17*19 = 9699690
    # Use a smaller primorial for denser coverage: 2*3*5*7*11*13 = 30030
    primorial = 30030
    window_size = 200
    
    # Phase 1: Search along arithmetic progression (multiples of primorial)
    # This ensures numbers are coprime to small primes, reducing collision probability
    if time.time() - start_time < LIMIT - 30:
        # Generate candidate starting points: k * primorial for k in range
        # We want s >= 1 and s + k - 1 <= 10^12
        max_k_val = (MAX_S - window_size + 1) // primorial
        # Sample a subset of k values to keep time bounded
        num_ap_samples = 2000
        if max_k_val > num_ap_samples:
            # Pick evenly spaced samples
            k_indices = np.linspace(1, max_k_val, num_ap_samples, dtype=np.int64)
        else:
            k_indices = np.arange(1, max_k_val + 1, dtype=np.int64)
        
        for k_val in k_indices:
            if time.time() - start_time > LIMIT - 30:
                break
            
            s = int(k_val * primorial)
            if s < 1 or s + window_size - 1 > MAX_S:
                continue
            
            phis = compute_phi_segment(s, s + window_size - 1)
            max_len, max_start_idx = find_longest_run(phis, window_size)
            
            if max_len > best_k:
                best_k = max_len
                best_s = s + max_start_idx
                
                if best_k >= 200:
                    break
    
    # Phase 2: Random sampling to find promising regions
    random.seed(42)
    num_samples = 3000
    for _ in range(num_samples):
        if time.time() - start_time > LIMIT - 20:
            break
        
        s = random.randint(1, MAX_S - window_size + 1)
        phis = compute_phi_segment(s, s + window_size - 1)
        max_len, max_start_idx = find_longest_run(phis, window_size)
        
        if max_len > best_k:
            best_k = max_len
            best_s = s + max_start_idx
            
            if best_k >= 200:
                break
    
    # Phase 3: Local refinement around best found region
    if best_k < 200 and time.time() - start_time < LIMIT - 10:
        expand_size = 2000
        s_start = max(1, best_s - expand_size)
        s_end = min(MAX_S, best_s + expand_size)
        
        if s_end - s_start + 1 > 0:
            # Process in chunks to avoid memory issues
            chunk_size = 5000
            for chunk_start in range(s_start, s_end + 1, chunk_size):
                if time.time() - start_time > LIMIT - 10:
                    break
                chunk_end = min(chunk_start + chunk_size - 1, s_end)
                phis = compute_phi_segment(chunk_start, chunk_end)
                max_len, max_start_idx = find_longest_run(phis, 200)
                
                if max_len > best_k:
                    best_k = max_len
                    best_s = chunk_start + max_start_idx
    
    # Final verification
    if best_k > 1:
        verify_phis = compute_phi_segment(best_s, best_s + best_k - 1)
        if len(set(verify_phis)) != best_k:
            # Fallback to a valid shorter run
            max_len, max_start_idx = find_longest_run(verify_phis, best_k)
            best_k = max_len
            best_s = best_s + max_start_idx
    
    return {'start': best_s, 'length': best_k}
