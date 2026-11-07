import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # We will search up to this limit for k
    # With O(N) sieve, we can afford a larger N.
    # 100 million is a good target if memory allows (arrays of int64 are 800MB each, 
    # we need sigma (800MB) and maybe a temp array. 1.6GB total might be tight but feasible.
    # Let's try 50 million first to be safe on memory, then maybe 100M if optimized.
    # Actually, we can do it in place or with minimal extra memory.
    N_MAX = 50000000
    
    # Precompute sigma for all k using a multiplicative sieve
    # sigma[1] = 1
    # For prime p, sigma[p] = 1 + p
    # For p^k, sigma[p^k] = (p^(k+1)-1)/(p-1)
    
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    sigma_arr[1] = 1
    
    # smallest_prime_factor (spf) or just a flag for primes?
    # We need to know the highest power of the smallest prime factor.
    # Let's use an array 'exp' to store the exponent of the smallest prime factor, 
    # or better, just iterate primes and update multiples.
    
    # A more standard O(N) approach:
    # Maintain an array 'sigma' and 'spow' (smallest prime power) and 'sprime' (smallest prime).
    # Or simply:
    # Initialize sigma[i] = 1 for all i? No.
    # Let's use the linear sieve for sigma.
    
    # sigma[1] = 1
    # For i from 2 to N:
    #   if not visited[i]: primes.append(i), sigma[i] = 1 + i, spow[i] = i, sprime[i] = i
    #   for p in primes:
    #     if i * p > N: break
    #     if i % p == 0:
    #       spow[i*p] = spow[i] * p
    #       sigma[i*p] = sigma[i] / (1 + spow[i]/p) * (1 + spow[i*p]/p) ... wait, simpler:
    #       sigma[i*p] = sigma[i] * (1 + p) - spow[i] * p ... no.
    #       Formula: if i = p^k * m, gcd(m,p)=1. sigma(i) = sigma(p^k)*sigma(m).
    #       sigma(p^(k+1)) = sigma(p^k) + p^(k+1).
    #       So if i % p == 0, let spow[i] be p^k. Then spow[i*p] = spow[i]*p.
    #       sigma[i*p] = sigma[i] - spow[i] + spow[i]*p ? No.
    #       sigma[i] = sigma[i/p^k] * sigma[p^k].
    #       sigma[i*p] = sigma[i/p^k] * sigma[p^(k+1)].
    #       sigma[p^(k+1)] = sigma[p^k] + p^(k+1).
    #       So sigma[i*p] = sigma[i] + sigma[i/p^k] * p^(k+1) ... complicated.
        
    # Let's use the simpler "additive" sieve but optimized with NumPy?
    # The original O(N log N) was the bottleneck.
    # Can we do a faster additive sieve?
    # No, multiplicative is O(N).
    
    # Let's implement the O(N) multiplicative sieve carefully.
    # Arrays:
    # sigma: int64
    # spow: int64 (largest power of smallest prime factor)
    # sprime: int64 (smallest prime factor)
    # is_prime: bool
    
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    spow_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    sprime_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    sigma_arr[1] = 1
    spow_arr[1] = 1
    sprime_arr[1] = 1
    
    primes = []
    is_prime = np.ones(N_MAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # This loop is O(N) but Python loop overhead is high.
    # However, 50M iterations in Python might take ~5-10 seconds? 
    # 100M might take 10-20s. This is much better than 25M O(N log N) which is ~100s+.
    
    for i in range(2, N_MAX + 1):
        if time.time() > deadline - 20:
            N_MAX = i
            break
            
        if is_prime[i]:
            primes.append(i)
            sigma_arr[i] = 1 + i
            spow_arr[i] = i
            sprime_arr[i] = i
            
        for p in primes:
            if i * p > N_MAX:
                break
            is_prime[i * p] = False
            if i % p == 0:
                spow_arr[i * p] = spow_arr[i] * p
                # sigma[i*p] = sigma[i] + spow_arr[i*p] / p * (1 + p) ... 
                # Actually: sigma[i*p] = sigma[i] + spow_arr[i*p]
                # Let's verify:
                # i = p^k * m. spow[i] = p^k. sigma[i] = sigma(m) * (1 + p + ... + p^k).
                # i*p = p^(k+1) * m. spow[i*p] = p^(k+1).
                # sigma[i*p] = sigma(m) * (1 + p + ... + p^(k+1)).
                # sigma[i*p] - sigma[i] = sigma(m) * p^(k+1) = (sigma[i] / (1+...+p^k)) * p^(k+1).
                # This doesn't look like a simple addition unless sigma(m)=1?
                # Wait, sigma(p^k) = (p^(k+1)-1)/(p-1).
                # sigma(p^(k+1)) = (p^(k+2)-1)/(p-1).
                # Difference is p^(k+1).
                # So sigma[i*p] = sigma[i] + sigma(m) * p^(k+1).
                # We don't have sigma(m) stored directly.
                
                # Alternative formula:
                # sigma[i*p] = sigma[i] * (1 + p) - spow_arr[i] * p ?
                # Let i = p^k * m.
                # sigma[i] = sigma(m) * S_k where S_k = 1+...+p^k.
                # sigma[i*p] = sigma(m) * S_{k+1} = sigma(m) * (S_k + p^(k+1)).
                # sigma[i] * (1+p) = sigma(m) * S_k * (1+p).
                # S_k * (1+p) = (1+...+p^k)(1+p) = 1 + 2p + ... + p^k + p^(k+1).
                # This is not S_{k+1}.
                
                # Correct recurrence:
                # If i % p == 0:
                #   spow[i*p] = spow[i] * p
                #   sigma[i*p] = sigma[i] + spow[i*p] // p * (1 + p) ... no.
                
                # Let's stick to the definition:
                # sigma[n] = sum of divisors.
                # If n = p^k * m, gcd(m,p)=1.
                # sigma[n] = sigma[m] * sigma[p^k].
                # Let spow[n] = p^k.
                # Then sigma[n] = sigma[n // spow[n]] * (spow[n]*p - 1)//(p-1).
                # This requires division which is slow.
                
                # Better:
                # sigma[i*p] = sigma[i] + spow[i*p]
                # Let's check:
                # i = p^k * m. spow[i] = p^k.
                # sigma[i] = sigma[m] * (p^(k+1)-1)/(p-1).
                # i*p = p^(k+1) * m. spow[i*p] = p^(k+1).
                # sigma[i*p] = sigma[m] * (p^(k+2)-1)/(p-1).
                # sigma[i*p] - sigma[i] = sigma[m] * [ (p^(k+2)-1) - (p^(k+1)-1) ] / (p-1)
                # = sigma[m] * p^(k+1) / (p-1) ... wait.
                # (p^(k+2)-1) - (p^(k+1)-1) = p^(k+2) - p^(k+1) = p^(k+1)(p-1).
                # So difference is sigma[m] * p^(k+1).
                # Is sigma[m] * p^(k+1) equal to spow[i*p]?
                # spow[i*p] = p^(k+1).
                # So difference is sigma[m] * spow[i*p].
                # We don't have sigma[m].
                
                # Let's use the property:
                # sigma[i*p] = sigma[i] + spow[i*p] * (sigma[i] // sigma[i//spow[i]]) ... no.
                
                # Okay, let's use the standard linear sieve for sigma from competitive programming:
                # sigma[1] = 1
                # for i in 2..N:
                #   if prime[i]:
                #     primes.append(i)
                #     sigma[i] = 1 + i
                #     spow[i] = i
                #   for p in primes:
                #     if i*p > N: break
                #     if i % p == 0:
                #       spow[i*p] = spow[i] * p
                #       sigma[i*p] = sigma[i] + spow[i*p] // p * (1 + p) ... 
                #       Actually, many sources say:
                #       sigma[i*p] = sigma[i] + spow[i*p]
                #       Let's test i=4, p=2.
                #       i=4 (2^2). spow[4]=4. sigma[4]=1+2+4=7.
                #       i*p=8 (2^3). spow[8]=8.
                #       sigma[8]=1+2+4+8=15.
                #       sigma[4] + spow[8] = 7 + 8 = 15. Correct!
                #       Let's test i=6 (2*3), p=2.
                #       i=6. spow[6]=2 (smallest prime is 2, exp 1). sigma[6]=1+2+3+6=12.
                #       i*p=12 (4*3). spow[12]=4.
                #       sigma[12]=1+2+3+4+6+12=28.
                #       sigma[6] + spow[12] = 12 + 4 = 16. Incorrect!
                
                # Why?
                # i=6 = 2^1 * 3. spow[6]=2.
                # sigma[6] = sigma[3] * sigma[2] = 4 * 3 = 12.
                # i*p=12 = 2^2 * 3. spow[12]=4.
                # sigma[12] = sigma[3] * sigma[4] = 4 * 7 = 28.
                # Difference = 16.
                # spow[12] = 4.
                # sigma[i] + spow[i*p] = 12 + 4 = 16.
                # It worked for prime powers because sigma[m]=1.
                # For general i, difference is sigma[m] * p^(k+1).
                # We need sigma[m].
                # m = i // spow[i].
                # So sigma[i*p] = sigma[i] + sigma[i // spow[i]] * spow[i*p].
                # This requires accessing sigma[i // spow[i]].
                
                sigma_i_div_spow = sigma_arr[i // spow_arr[i]]
                sigma_arr[i * p] = sigma_arr[i] + sigma_i_div_spow * spow_arr[i * p]
                break
            else:
                spow_arr[i * p] = p
                sigma_arr[i * p] = sigma_arr[i] * (1 + p)
                
    # Ensure we have valid arrays up to N_MAX
    # Compute M = k * sigma(k)
    k_vals = np.arange(1, N_MAX + 1, dtype=np.int64)
    # Filter out M > 10^18
    # M = k * sigma[k]
    # We can compute M for all, then filter.
    # But k * sigma[k] might overflow int64?
    # k <= 5e7. sigma[k] <= k * log(log k) approx.
    # Max sigma is around 1e8.
    # 5e7 * 1e8 = 5e15 < 9e18 (int64 max). Safe.
    
    M_vals = k_vals * sigma_arr[1:N_MAX+1]
    
    # Mask valid M <= 10^18
    mask = M_vals <= 10**18
    M_valid = M_vals[mask]
    k_valid = k_vals[mask]
    
    # Find duplicates
    # Use np.unique to count occurrences
    # This is O(N log N) in C, which is fast.
    unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
    
    # Find M with max count
    if len(unique_M) == 0:
         return {'M': 333312, 'k': [336, 372, 434]}
         
    max_count = np.max(counts)
    
    # If max_count is 1, no collisions found? 
    # The problem asks for distinct k_i. 
    # If max_count is 1, we can't form a set of >1 solutions? 
    # The score is the number of listed solutions. 
    # If we list 1 solution, score is 1.
    # But we want to maximize.
    # If no M has >1 k, we return the best single one? 
    # The task says "k is a list of ... distinct ints ... satisfying ...".
    # It doesn't explicitly say len(k) > 1, but "many solutions" implies >1.
    # However, if max_count is 1, we just pick one.
    
    # Get indices of M with max count
    # We want the M itself
    best_M_idx = np.argmax(counts)
    best_M = int(unique_M[best_M_idx])
    
    # Get all k that map to best_M
    # inverse[i] is the index in unique_M for M_valid[i]
    # We want k_valid where inverse == best_M_idx
    # This is a boolean mask operation.
    k_indices = np.where(inverse == best_M_idx)[0]
    best_ks = k_valid[k_indices]
    
    # Convert to Python ints
    final_ks = [int(k) for k in best_ks]
    
    # Verify
    def sigma_exact(n):
        if n == 1: return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    p_power *= p
                result *= (p_power * p - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    verified_ks = []
    for k in final_ks:
        if k * sigma_exact(k) == best_M:
            verified_ks.append(k)
            
    if not verified_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': verified_ks}

