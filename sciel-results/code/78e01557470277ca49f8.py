import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # We want to find M with many k such that k * sigma(k) = M.
    # We use a linear sieve to compute sigma(k) for k up to N.
    
    # Estimate N based on time.
    # Linear sieve is O(N). 
    # Let's try N = 5,000,000 first.
    N = 5000000
    
    # Arrays for linear sieve
    # sigma[n]
    # spf[n] - smallest prime factor
    # exp[n] - exponent of spf in n
    # val[n] - spf[n] ^ exp[n]
    # rest_sigma[n] - sigma(n / val[n])
    
    # Actually, simpler:
    # sigma[n]
    # lp[n] - largest prime factor? No, smallest is standard.
    # Let's use:
    # sigma[n]
    # p_pow[n] = p^a where p is spf(n) and a is its exponent in n.
    # sigma_rest[n] = sigma(n / p^a)
    
    sigma = [0] * (N + 1)
    p_pow = [0] * (N + 1)
    sigma_rest = [0] * (N + 1)
    spf = [0] * (N + 1) # smallest prime factor
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_rest[1] = 1
    
    primes = []
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            p_pow[i] = i
            sigma_rest[i] = 1
            
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                # p divides i
                # i = p^a * m, gcd(p,m)=1
                # i*p = p^(a+1) * m
                # p_pow[i*p] = p_pow[i] * p
                # sigma(i*p) = sigma(i) + p^(a+1) * sigma(m)
                # sigma(m) = sigma_rest[i]
                # p^(a+1) = p_pow[i] * p
                p_pow[i * p] = p_pow[i] * p
                sigma_rest[i * p] = sigma_rest[i]
                sigma[i * p] = sigma[i] + p_pow[i] * p * sigma_rest[i]
                break
            else:
                # p does not divide i
                # i*p = p * i, gcd(p,i)=1
                # p_pow[i*p] = p
                # sigma_rest[i*p] = sigma(i)
                # sigma(i*p) = sigma(i) * (1+p)
                p_pow[i * p] = p
                sigma_rest[i * p] = sigma[i]
                sigma[i * p] = sigma[i] * (1 + p)
                
        if time.time() > deadline - 10.0:
            break

    # If we broke early, N is effectively smaller
    # We need to find the best group
    groups = {}
    max_k_processed = 1
    for k in range(1, N + 1):
        if sigma[k] == 0:
            continue
        c = k * sigma[k]
        if c not in groups:
            groups[c] = []
        groups[c].append(k)
        max_k_processed = k
        if time.time() > deadline - 5.0:
            break

    # Find best group
    best_group = []
    best_c = 0
    for c, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_c = c
            
    if len(best_group) >= 3:
        M = int(best_c)
        # Verify k <= sqrt(M)
        sqrt_M = math.isqrt(M)
        valid_ks = [int(x) for x in best_group if x <= sqrt_M]
        if len(valid_ks) >= 3:
            return {'M': M, 'k': valid_ks}
            
    # Fallback to known good solution
    return {'M': 333312, 'k': [336, 372, 434]}
