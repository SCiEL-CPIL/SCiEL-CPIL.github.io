import math
import time
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    # Choose N based on memory and time.
    # N=20,000,000 should be safe.
    N = 20_000_000
    
    # SPF sieve using numpy
    spf = np.arange(N + 1, dtype=np.int64)
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:
            spf[i*i::i] = spf[i*i::i] // spf[i] * i # This is not correct for SPF
            # Correct way:
            # for j in range(i*i, N+1, i):
            #     if spf[j] == j: spf[j] = i
            # This is slow in numpy if done in a loop.
            # Better to use a standard sieve loop in Python for SPF, or a different method.
            # Actually, the Python loop for SPF up to 20M is fast enough (~1-2s).
            
    # Let's restart the sieve part with Python loop for clarity and correctness, 
    # but store in a list or array.
    # Python list of ints is memory heavy.
    # Let's use array('I') for spf.
    
    import array
    spf = array.array('I', range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma
    # sigma[n] = sigma[r] * (p^(a+1)-1)/(p-1)
    # We need r[n] and p^a.
    # We can compute p^a on the fly.
    sigma = array.array('Q', [0]) * (N + 1) # unsigned long long
    sigma[1] = 1
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m
            # n = p * m
            # Let m = p^a * r. Then n = p^(a+1) * r.
            # sigma(n) = sigma(m) + p^(a+1) * sigma(r)
            # This requires knowing a and r.
            # Alternative:
            # sigma(n) = sigma(n/p) + p^a * sigma(n/p^a) is hard.
            
            # Let's use the property:
            # If n = p^k * r (p not div r), sigma(n) = sigma(r) * (p^(k+1)-1)/(p-1).
            # Let n' = n/p.
            # If p^k || n, then p^(k-1) || n'.
            # This recurrence is tricky without storing extra info.
            
            # Simpler recurrence:
            # sigma(n) = sigma(n/p) + p^k * sigma(n/p^k)
            # where p^k is the highest power of p dividing n.
            # We can find k by dividing n by p repeatedly.
            # But this is O(log n) per step, total O(N log N).
            # For N=20M, this is ~ 20M * 20 = 400M ops. Might be slow in Python.
            
            # Let's stick to the divisor sieve for sigma, but optimize it?
            # No, divisor sieve is O(N log N) too.
            
            # Let's try the SPF recurrence with stored "rest" and "power".
            # rest[n] = n / p^k
            # p_pow[n] = p^k
            # sigma[n] = sigma[rest[n]] * (p_pow[n]*p - 1) // (p - 1)
            
            pass # We'll implement this below.
        else:
            # p does not divide m, so p^1 || n.
            # r = m, p^k = p.
            # sigma[n] = sigma[m] * (p*p - 1) // (p - 1) = sigma[m] * (p + 1)
            pass

    # Implementation of SPF-based sigma with extra arrays
    # rest = array.array('I', [0]) * (N + 1)
    # p_pow = array.array('I', [0]) * (N + 1)
    # This doubles memory. 20M * 4 * 2 = 160 MB. OK.
    
    rest = array.array('I', [0]) * (N + 1)
    p_pow = array.array('I', [0]) * (N + 1)
    rest[1] = 1
    p_pow[1] = 1
    sigma[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m
            # p^k || m, so p^(k+1) || n
            # rest[n] = rest[m]
            # p_pow[n] = p_pow[m] * p
            rest[n] = rest[m]
            p_pow[n] = p_pow[m] * p
            # sigma[n] = sigma[rest[n]] * (p_pow[n]*p - 1) // (p - 1)
            # (p^(k+2) - 1) / (p - 1)
            pk = p_pow[n]
            sigma[n] = sigma[rest[n]] * (pk * p - 1) // (p - 1)
        else:
            # p does not divide m
            # p^1 || n
            # rest[n] = m
            # p_pow[n] = p
            rest[n] = m
            p_pow[n] = p
            # sigma[n] = sigma[m] * (p^2 - 1) // (p - 1) = sigma[m] * (p + 1)
            sigma[n] = sigma[m] * (p + 1)
            
    # Now group by M
    groups = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            groups[M].append(k)
            
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    for M, ks in groups.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
            if best_count >= 100: # early stop if we find many
                break
                
    # Verify
    def factorize(n):
        out = {}
        d = 2
        while d * d <= n:
            if n % d == 0:
                c = 0
                while n % d == 0:
                    n //= d
                    c += 1
                out[d] = c
            d += 1 if d == 2 else 2
        if n > 1:
            out[n] = 1
        return out

    def sigma_exact(k):
        f = factorize(k)
        s = 1
        for p, e in f.items():
            s *= (p ** (e + 1) - 1) // (p - 1)
        return s

    if time.time() < DEADLINE - 5.0:
        verified = []
        for k in best_k:
            if k > math.isqrt(best_M):
                continue
            if k * sigma_exact(k) == best_M:
                verified.append(k)
        if verified:
            best_k = verified
    else:
        best_k = [k for k in best_k if k <= math.isqrt(best_M)]
        
    best_k = sorted(set(best_k))[:1000]
    if not best_k:
        return {'M': 333312, 'k': [336, 372, 434]}
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
