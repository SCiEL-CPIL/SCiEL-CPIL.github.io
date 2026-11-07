"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def _sigma_linear_sieve(n):
    """Compute sigma(k) for all k from 1 to n using a linear sieve.
    
    Uses the multiplicative property of sigma:
    - If p | n and p^2 | n: sigma(n) = sigma(n/p) + p * sigma(n/p^2) ... 
      Actually simpler: sigma is multiplicative. For prime powers:
      sigma(p^a) = (p^(a+1)-1)/(p-1)
    
    Linear sieve approach:
    - sigma[1] = 1
    - For each number, track its smallest prime factor and the power of that prime
    """
    sigma = [0] * (n + 1)
    sigma[1] = 1
    is_composite = [False] * (n + 1)
    primes = []
    # lp[i] = smallest prime factor of i
    # sp[i] = the largest power of lp[i] dividing i (i.e., p^a where p^a || i)
    # rest[i] = i / sp[i]
    lp = [0] * (n + 1)
    sp = [1] * (n + 1)  # the prime power part
    rest = [1] * (n + 1)  # the coprime remainder
    
    for i in range(2, n + 1):
        if not is_composite[i]:
            primes.append(i)
            lp[i] = i
            sp[i] = i
            rest[i] = 1
            sigma[i] = 1 + i  # sigma(p) = 1 + p
        
        for p in primes:
            if p > lp[i] or i * p > n:
                break
            is_composite[i * p] = True
            lp[i * p] = p
            if p == lp[i]:
                # p divides i, so p^(a+1) divides i*p
                sp[i * p] = sp[i] * p
                rest[i * p] = rest[i]
                # sigma(i*p) = sigma(sp[i]*p) * sigma(rest[i])
                # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
                sigma[i * p] = (sigma[sp[i]] + sp[i] * p) * sigma[rest[i]]
                break
            else:
                # p does not divide i, so p is coprime to i
                sp[i * p] = p
                rest[i * p] = i
                sigma[i * p] = sigma[i] * (1 + p)
    
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    from collections import defaultdict
    
    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try increasing sizes with the linear sieve
    sizes = [500000, 1000000, 2000000, 5000000, 10000000]
    
    for N in sizes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:  # need at least 20s for scanning
            break
        
        sigma = _sigma_linear_sieve(N)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        m_to_ks = defaultdict(list)
        
        for k in range(1, N + 1):
            m = k * sigma[k]
            if m <= 10**18 and k <= math.isqrt(m):
                m_to_ks[m].append(k)
        
        for m, ks in m_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
    
    # Ensure types are correct
    best_m = int(best_m)
    best_ks = list(set(int(k) for k in best_ks))
    best_ks.sort()
    
    return {'M': best_m, 'k': best_ks}
# EVOLVE-BLOCK-END
