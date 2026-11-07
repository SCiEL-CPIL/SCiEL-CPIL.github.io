import math
import time
from array import array
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin

    # 1. Linear Sieve for k up to N
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
        # If sieve took too long, just use what we have
        best_group = []
        best_M = 0
        for c, xs in groups.items():
            if len(xs) > len(best_group):
                best_group = xs
                best_M = c
        if len(best_group) >= 3:
            M = int(best_M)
            sqrt_M = math.isqrt(M)
            # Use a set to ensure distinctness, though sieve k's are unique
            valid_ks = list(set(int(x) for x in best_group if x <= sqrt_M))
            if len(valid_ks) >= 3:
                return {'M': M, 'k': valid_ks}
        return {'M': 333312, 'k': [336, 372, 434]}

    # 2. Smooth Number Search (DFS)
    # Use primes up to 31 to balance depth and breadth
    smooth_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    max_M = 10**18
    
    # Iterative DFS using a stack to avoid recursion limits/overhead if needed, 
    # but recursion depth 11 is safe and fast.
    def dfs(i, current_k, sigma_prod):
        if time.time() > deadline - 5.0:
            return

        if i == len(smooth_primes):
            if current_k > 1:
                M = current_k * sigma_prod
                if M <= max_M:
                    groups[M].append(current_k)
            return
        
        p = smooth_primes[i]
        
        # Option 1: Exponent 0
        dfs(i + 1, current_k, sigma_prod)
        
        # Option 2: Exponent e >= 1
        k_curr = current_k
        sig_p = 1 # sum of powers 1 + p + ... + p^e
        p_power = 1
        
        while True:
            k_curr *= p
            p_power *= p
            sig_p += p_power
            
            # Check overflow / limit
            if k_curr > max_M // sigma_prod or sigma_prod > max_M // sig_p:
                break
            
            M_val = k_curr * (sigma_prod * sig_p)
            if M_val > max_M:
                break
                
            dfs(i + 1, k_curr, sigma_prod * sig_p)

    dfs(0, 1, 1)

    # 3. Find best group
    best_group = []
    best_M = 0
    for M, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_M = M
            
    if len(best_group) >= 3:
        M = int(best_M)
        sqrt_M = math.isqrt(M)
        # Ensure distinctness and validity
        valid_ks = list(set(int(x) for x in best_group if x <= sqrt_M))
        if len(valid_ks) >= 3:
            return {'M': M, 'k': valid_ks}
            
    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
