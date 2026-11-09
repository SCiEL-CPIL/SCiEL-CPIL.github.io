import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin (150s kill, leave buffer)
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Candidate N values - use larger range with faster sieve
    N_candidates = [5 * 10**6, 10 * 10**6, 15 * 10**6, 20 * 10**6]
    
    for N in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        # Linear sieve for sigma computation
        # sigma[1] = 1
        # For prime p: sigma(p) = p + 1
        # For p^a || n: sigma(p^a) = (p^(a+1)-1)/(p-1)
        # sigma is multiplicative
        
        sigma = np.zeros(N + 1, dtype=np.int64)
        sigma[1] = 1
        
        # Linear sieve arrays
        is_prime = np.ones(N + 1, dtype=bool)
        primes = []
        smallest_prime = np.zeros(N + 1, dtype=np.int64)
        # sigma_prime_power[i] = sigma of the highest power of smallest_prime[i] dividing i
        # Actually, let's use a different approach
        
        # Use the formula: sigma(n) = product of (p^(a+1)-1)/(p-1) for p^a || n
        # Linear sieve: maintain for each n:
        # - spf[n]: smallest prime factor
        # - spf_power[n]: highest power of spf dividing n
        # - rest_sigma[n]: sigma(n / spf_power[n])
        
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
                    # sigma(i*p) = sigma(spf_power[i*p]) * rest_sigma[i*p]
                    # sigma(p^(a+1)) = sigma(p^a) * p + 1... no
                    # sigma(p^a) = (p^(a+1)-1)/(p-1)
                    # sigma(p^(a+1)) = sigma(p^a) * p + 1? No.
                    # sigma(p^a) = 1 + p + ... + p^a
                    # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
                    # So sigma(i*p) = (sigma(spf_power[i*p]) - sigma(spf_power[i]) + spf_power[i*p]) * rest_sigma[i*p]
                    # = (sigma(spf_power[i]) + spf_power[i*p]) * rest_sigma[i*p]
                    # = sigma(spf_power[i]) * rest_sigma[i*p] + spf_power[i*p] * rest_sigma[i*p]
                    # But sigma(i) = sigma(spf_power[i]) * rest_sigma[i]
                    # So sigma(i*p) = sigma(i) + spf_power[i*p] * rest_sigma[i*p]
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
        
        # Find M with most k
        M_to_k = defaultdict(list)
        for k in range(1, N + 1):
            M = k * int(sigma[k])
            if M <= 10**18:
                M_to_k[M].append(k)
        
        # Find best
        current_best_M = None
        current_best_k_list = None
        max_count = 0
        for M, k_list in M_to_k.items():
            if len(k_list) > max_count:
                max_count = len(k_list)
                current_best_M = M
                current_best_k_list = k_list
        
        if current_best_M is not None:
            if len(current_best_k_list) > len(best_k):
                best_M = current_best_M
                best_k = current_best_k_list
                
    return {'M': best_M, 'k': best_k}
