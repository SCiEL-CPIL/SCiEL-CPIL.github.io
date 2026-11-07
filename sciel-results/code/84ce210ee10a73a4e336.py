import math
import time
from collections import defaultdict

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 140.0

    def _fallback():
        # Try several known M values with multiple solutions
        candidates = [
            (333312, [336, 372, 434]),
            (1, [1]),
        ]
        for M, ks in candidates:
            ok = []
            for k in ks:
                if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                    ok.append(k)
            if len(ok) >= 1:
                return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    import numpy as np

    # Use linear sieve for sigma computation - O(N) instead of O(N log N)
    # sigma(n) = sum of divisors of n
    # Using the property: if n = p^a * m where p doesn't divide m,
    # sigma(n) = sigma(p^a) * sigma(m)
    
    # Try different sizes based on available time
    sizes_to_try = [30_000_000, 20_000_000, 15_000_000, 10_000_000, 5_000_000]
    
    best_result = None
    
    for N in sizes_to_try:
        if time.time() > deadline - 30.0:
            break
        
        try:
            # Linear sieve for sigma
            sigma = np.zeros(N + 1, dtype=np.int64)
            sigma[1] = 1
            primes = []
            is_composite = np.zeros(N + 1, dtype=bool)
            # For linear sieve, we need:
            # lp[n] = smallest prime factor
            # exp[n] = exponent of smallest prime factor in n
            # rest[n] = n with smallest prime factor removed
            # sigma[n] = sigma(rest[n]) * (1 + p + p^2 + ... + p^exp)
            
            # Actually, let's use a simpler approach: modified sieve
            # sigma[n] = sigma[n] + d for each divisor d of n
            # But that's O(N log N). Let's use the multiplicative property.
            
            # Linear sieve approach:
            # spf[n] = smallest prime factor
            # sigma[n] computed using multiplicative property
            
            spf = np.zeros(N + 1, dtype=np.int64)
            spf[1] = 1
            sigma_arr = np.zeros(N + 1, dtype=np.int64)
            sigma_arr[1] = 1
            primes_list = []
            
            for i in range(2, N + 1):
                if spf[i] == 0:
                    spf[i] = i
                    primes_list.append(i)
                    sigma_arr[i] = 1 + i  # sigma(p) = 1 + p for prime p
                
                for p in primes_list:
                    if p > spf[i] or i * p > N:
                        break
                    spf[i * p] = p
                    if p == spf[i]:
                        # p divides i, so sigma(i*p) = sigma(i) * (1 + p) if i is not divisible by p
                        # Actually need to be more careful
                        # Let i = p^k * m where p doesn't divide m
                        # sigma(i) = (1+p+...+p^k) * sigma(m)
                        # sigma(i*p) = (1+p+...+p^(k+1)) * sigma(m)
                        # = sigma(i) * (1+p+...+p^(k+1))/(1+p+...+p^k)
                        # This is tricky. Let's use a different approach.
                        break
                    else:
                        # p doesn't divide i
                        sigma_arr[i * p] = sigma_arr[i] * (1 + p)
                
                if time.time() > deadline - 30.0:
                    break
            
            # The linear sieve above is incomplete. Let me use a better approach.
            # Actually, let's just use the standard sieve but with NumPy optimization
            
            # Reset and use the original approach but optimized
            sigma = np.zeros(N + 1, dtype=np.int64)
            for d in range(1, N + 1):
                sigma[d::d] += d
                if time.time() > deadline - 30.0:
                    break
            
            if time.time() > deadline - 25.0:
                continue
            
            # Build map from M -> list of k
            M_to_ks = defaultdict(list)
            sigma_list = sigma.tolist()
            for k in range(1, N + 1):
                M_val = k * sigma_list[k]
                if M_val <= 10**18:
                    M_to_ks[M_val].append(k)
                if time.time() > deadline - 20.0:
                    break
            
            if time.time() > deadline - 15.0:
                continue
            
            # Pick the M with the most valid k's (k <= sqrt(M))
            best_M = None
            best_sols = None
            for M_val, ks in M_to_ks.items():
                if len(ks) < 2:
                    continue
                limit = math.isqrt(M_val)
                valid = [k for k in ks if k <= limit]
                if best_sols is None or len(valid) > len(best_sols):
                    best_sols = valid
                    best_M = M_val
                if time.time() > deadline - 10.0:
                    break
            
            if best_sols is not None and len(best_sols) >= 2:
                # Final exact verification
                verified = []
                for k in best_sols:
                    if 1 <= k <= math.isqrt(best_M) and k <= N:
                        if k * sigma_list[k] == best_M:
                            verified.append(k)
                
                if verified:
                    verified = sorted(set(verified))
                    if best_result is None or len(verified) > len(best_result['k']):
                        best_result = {'M': int(best_M), 'k': [int(x) for x in verified]}
            
            # If we found 5+ solutions, we can stop early
            if best_result is not None and len(best_result['k']) >= 5:
                break
                
        except Exception:
            continue
    
    if best_result is not None and len(best_result['k']) >= 2:
        return best_result
    
    return _fallback()
