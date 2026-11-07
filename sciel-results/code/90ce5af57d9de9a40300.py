import math
import time
from array import array
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Sieve for k up to N
    N = 20000000  # 2 * 10^7
    sigma = array('q', [0]) * (N + 1)
    spf = array('i', [0]) * (N + 1)
    
    sigma[1] = 1
    primes = []
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                # i = p^a * m, p|m is false
                # sigma(i*p) = sigma(i) + p * (sigma(i) - sigma(i/p)) ?
                # Standard linear sieve update:
                # if i % p == 0:
                #   sigma(i*p) = sigma(i) + i*p
                # else:
                #   sigma(i*p) = sigma(i) * (1 + p)
                if i % p == 0:
                    sigma[i * p] = sigma[i] + i * p
                    break
                else:
                    sigma[i * p] = sigma[i] * (1 + p)
                    
        if time.time() > deadline - 20.0:
            break

    groups = defaultdict(list)
    
    # Process sieve results
    limit_k = N
    for k in range(1, limit_k + 1):
        s = sigma[k]
        if s == 0: continue
        M = k * s
        if M <= 10**18:
            groups[M].append(k)
            
    if time.time() > deadline - 10.0:
        best_group = []
        for c, xs in groups.items():
            if len(xs) > len(best_group):
                best_group = xs
        if len(best_group) >= 3:
            M = int(list(groups.keys())[0]) # placeholder
            # find M
            for c, xs in groups.items():
                if xs == best_group:
                    M = int(c)
                    break
            sqrt_M = math.isqrt(M)
            valid_ks = [int(x) for x in best_group if x <= sqrt_M]
            if len(valid_ks) >= 3:
                return {'M': M, 'k': valid_ks}
        return {'M': 333312, 'k': [336, 372, 434]}

    # 2. Smooth Number Search
    # Primes to consider
    smooth_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    max_k_smooth = 500000000  # 5 * 10^8
    max_M = 10**18
    
    # DFS to generate smooth numbers
    # To avoid recursion depth issues and overhead, we can use a stack or just recursion with sys.setrecursionlimit
    # Depth is len(smooth_primes) = 11, so recursion is fine.
    
    def dfs(i, current_k, sigma_prod):
        nonlocal groups
        if i == len(smooth_primes):
            M = current_k * sigma_prod
            if M <= max_M:
                groups[M].append(current_k)
            return
        
        p = smooth_primes[i]
        k_p = 1
        sig_p = 1
        
        # We need to iterate exponents e such that current_k * p^e <= max_k_smooth
        while current_k * k_p <= max_k_smooth:
            new_k = current_k * k_p
            new_sigma_prod = sigma_prod * sig_p
            
            # Pruning: if new_k * new_sigma_prod > max_M, 
            # increasing k_p will only increase M, so break?
            # Yes, k_p increases, new_k increases, new_sigma_prod increases.
            if new_k * new_sigma_prod > max_M:
                break
            
            dfs(i + 1, new_k, new_sigma_prod)
            
            # Update for next exponent
            k_p *= p
            sig_p = sig_p * p + 1
            
            if time.time() > deadline - 5.0:
                return

    dfs(0, 1, 1)

    # Find best group
    best_group = []
    best_M = 0
    for M, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_M = M
            
    if len(best_group) >= 3:
        M = int(best_M)
        sqrt_M = math.isqrt(M)
        valid_ks = [int(x) for x in best_group if x <= sqrt_M]
        if len(valid_ks) >= 3:
            return {'M': M, 'k': valid_ks}
            
    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
