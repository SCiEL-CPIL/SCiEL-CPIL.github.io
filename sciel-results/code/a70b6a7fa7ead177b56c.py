import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 135  # Safety margin (150s kill, leave buffer)
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Candidate N values - use larger range with faster sieve
    N_candidates = [2 * 10**6, 5 * 10**6, 10 * 10**6, 20 * 10**6, 40 * 10**6]
    
    for N in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
            
        # Linear sieve for sigma computation
        sigma = np.zeros(N + 1, dtype=np.int64)
        sigma[1] = 1
        
        # Linear sieve arrays
        is_prime = np.ones(N + 1, dtype=bool)
        primes = []
        smallest_prime = np.zeros(N + 1, dtype=np.int64)
        
        # spf[n]: smallest prime factor
        # spf_power[n]: highest power of spf dividing n
        # rest_sigma[n]: sigma(n / spf_power[n])
        
        spf = np.zeros(N + 1, dtype=np.int64)
        spf_power = np.zeros(N + 1, dtype=np.int64)
        rest_sigma = np.zeros(N + 1, dtype=np.int64)
        
        sigma[1] = 1
        spf[1] = 1
        spf_power[1] = 1
        rest_sigma[1] = 1
        
        for i in range(2, N + 1):
            if spf[i] == 0:
                # i is prime
                spf[i] = i
                spf_power[i] = i
                rest_sigma[i] = 1
                sigma[i] = 1 + i
                primes.append(i)
            
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                spf[i * p] = p
                if p == spf[i]:
                    spf_power[i * p] = spf_power[i] * p
                    rest_sigma[i * p] = rest_sigma[i]
                    # sigma(i*p) = sigma(i) + spf_power[i*p] * rest_sigma[i*p]
                    sigma[i * p] = sigma[i] + spf_power[i * p] * rest_sigma[i * p]
                    break
                else:
                    spf_power[i * p] = p
                    rest_sigma[i * p] = sigma[i]
                    sigma[i * p] = (1 + p) * rest_sigma[i * p]
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            continue
        
        # Find M with most k using vectorized operations
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        
        # Filter M <= 10^18
        mask = M_arr <= 10**18
        M_valid = M_arr[mask]
        k_valid = k_arr[mask]
        
        if len(M_valid) == 0:
            continue
        
        # Find M with most k using np.unique
        unique_M, inverse = np.unique(M_valid, return_inverse=True)
        counts = np.bincount(inverse)
        max_count = int(counts.max())
        
        if max_count <= len(best_k):
            continue
        
        # Find all M that achieve max_count
        best_Ms = unique_M[counts == max_count]
        
        # Pick the one with smallest M (to be safe with sqrt constraint)
        # Actually, we need k <= sqrt(M), so for a given M, k must be <= isqrt(M)
        # Since k * sigma(k) = M and sigma(k) >= k+1 for k >= 2, we have k*(k+1) <= M, so k <= ~sqrt(M)
        # Let's just pick the first one and verify
        
        current_best_M = int(best_Ms[0])
        
        # Get all k for this M
        mask_k = M_valid == current_best_M
        current_best_k_list = [int(k) for k in k_valid[mask_k]]
        
        # Verify k <= isqrt(M)
        import math
        sqrt_M = math.isqrt(current_best_M)
        current_best_k_list = [k for k in current_best_k_list if k <= sqrt_M]
        
        if len(current_best_k_list) > len(best_k):
            best_M = current_best_M
            best_k = current_best_k_list
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}
