"""Rewritten program: find M <= 10^18 with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _isqrt(x):
    return math.isqrt(x)

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial division factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = (d ** (power + 1) - 1) // (d - 1)
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _divisors(n):
    """Return sorted list of all divisors of n."""
    if n <= 0:
        return []
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs

def _linear_sigma_sieve(N):
    """Compute sigma(k) for k=1..N using a linear sieve. Returns list indexed 0..N."""
    sigma = [0] * (N + 1)
    sigma[1] = 1
    primes = []
    
    spf = [0] * (N + 1)
    p_pow = [1] * (N + 1)
    sigma_p = [1] * (N + 1)
    rest_sig = [1] * (N + 1)
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_p[1] = 1
    rest_sig[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            p_pow[i] = i
            sigma_p[i] = 1 + i
            rest_sig[i] = 1
            sigma[i] = 1 + i
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                p_pow[i * p] = p_pow[i] * p
                sigma_p[i * p] = sigma_p[i] + p_pow[i * p]
                rest_sig[i * p] = rest_sig[i]
            else:
                p_pow[i * p] = p
                sigma_p[i * p] = 1 + p
                rest_sig[i * p] = sigma[i]
            sigma[i * p] = sigma_p[i * p] * rest_sig[i * p]
    
    return sigma

def _numpy_sigma_sieve(N):
    """Compute sigma(k) for k=1..N using NumPy vectorized approach. Returns numpy array indexed 0..N."""
    import numpy as np
    
    sigma = np.ones(N + 1, dtype=np.int64)
    
    for p in range(2, N + 1):
        if sigma[p] == 1:  # p is prime (sigma(p) would be 1+p if computed, but we use a different check)
            # Actually, let's use a different approach: sieve for primes first
            pass
    
    # Better: use the standard sieve approach with NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # For each number, add its contribution to sigma of multiples
    # sigma(n) = sum of d | n of d
    # So for each d, add d to sigma[m] for all m that are multiples of d
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Phase 1: Use NumPy sieve for fast sigma computation
    import numpy as np
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try increasing sizes of N
    sizes_to_try = [2000000, 5000000, 10000000]
    
    for N in sizes_to_try:
        if time.time() > deadline - 30:
            break
        
        # Compute sigma using NumPy
        sigma = _numpy_sigma_sieve(N)
        
        # Compute M_val = k * sigma[k] for all k
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:N + 1]
        
        # Filter M_vals <= 10^18
        mask = M_vals <= 10**18
        valid_M_vals = M_vals[mask]
        valid_k_vals = k_vals[mask]
        
        # Find collisions: group by M_val
        # Use a dict to count occurrences
        m_to_k = defaultdict(list)
        for i in range(len(valid_M_vals)):
            m_to_k[int(valid_M_vals[i])].append(int(valid_k_vals[i]))
        
        # Find top candidates
        candidates = sorted(m_to_k.items(), key=lambda x: len(x[1]), reverse=True)
        top_candidates = candidates[:200]
        
        # For each top candidate, try to extend
        for M_val, ks in top_candidates:
            if time.time() > deadline - 15:
                break
            
            sqrt_M = _isqrt(M_val)
            valid_ks = [k for k in ks if k <= sqrt_M]
            current_count = len(valid_ks)
            
            # Skip if can't beat best even with generous extension
            if current_count + 10 <= best_count:
                continue
            
            # Try to extend: find divisors k of M_val with k > N, k <= sqrt_M
            divs = _divisors(M_val)
            existing = set(valid_ks)
            extended = list(valid_ks)
            
            for k in divs:
                if k <= N or k > sqrt_M or k in existing:
                    continue
                expected_sigma = M_val // k
                if k > 1 and expected_sigma < k + 1:
                    continue
                if _sigma_by_factorization(k) == expected_sigma:
                    extended.append(k)
                    existing.add(k)
            
            if len(extended) > best_count:
                best_M = M_val
                best_ks = sorted(extended)
                best_count = len(extended)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    # Final validation: verify all k satisfy k*sigma(k) == M and k <= sqrt(M)
    sqrt_M = _isqrt(best_M)
    final_ks = []
    for k in best_ks:
        if k <= sqrt_M and _sigma_by_factorization(k) * k == best_M:
            final_ks.append(k)
    
    return {'M': best_M, 'k': final_ks}

# EVOLVE-BLOCK-END
