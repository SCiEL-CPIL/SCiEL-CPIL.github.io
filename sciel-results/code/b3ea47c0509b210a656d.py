"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    MAX = 10000000
    
    # Build SPF (smallest prime factor) sieve
    spf = list(range(MAX + 1))
    for i in range(2, int(MAX**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Build sigma using multiplicative property via SPF
    # sigma(n) = product over prime powers p^k || n of (p^(k+1)-1)/(p-1)
    # We can compute this iteratively using spf
    sigma = [0] * (MAX + 1)
    sigma[1] = 1
    
    # Use a linear-ish approach: for each n, factor using spf
    # But that's O(n log n) total which might be slow in Python
    # Alternative: use the recurrence with spf
    
    # Better: compute sigma using the property that if n = p^k * m where p doesn't divide m,
    # then sigma(n) = sigma(p^k) * sigma(m)
    # We can do this with a sieve-like approach
    
    # Let's use: for each prime p, for each power p^k, multiply sigma contributions
    # Actually, let's just use the direct sieve but in a vectorized way with array operations
    
    # Most practical approach for Python: use the additive sieve but only up to a useful range
    # Or: compute sigma for all numbers using the formula with spf
    
    # Let me try the spf-based factorization approach for sigma
    # For each n, find its prime factorization using spf, compute sigma
    
    # This is O(n * log(n)) worst case but typically fast
    # Let's try a different approach: modified sieve
    
    # Actually, the fastest pure-Python approach for sigma up to 10^7:
    # Use the fact that sigma(n) = sum_{d|n} d
    # We can compute this with: for d in 1..N: for m in 1..N//d: sigma[d*m] += d
    # This is O(N log N) but with a small constant. In Python this is ~10^7 * H_10^7 ≈ 10^7 * 16 = 1.6*10^8
    # This is too slow in pure Python.
    
    # Let me use NumPy if available
    try:
        import numpy as np
        sigma_arr = np.zeros(MAX + 1, dtype=np.int64)
        for i in range(1, MAX + 1):
            sigma_arr[i::i] += i
        sigma = sigma_arr.tolist()
    except ImportError:
        # Fallback: compute sigma for a smaller range and use that
        # Or use the spf approach
        sigma = [0] * (MAX + 1)
        sigma[1] = 1
        # Use spf to compute sigma multiplicatively
        # For n > 1: let p = spf[n], n = p^k * m where p ∤ m
        # sigma[n] = sigma[p^k] * sigma[m]
        # We need to track the p-power part
        
        # Let's use arrays to track the prime power contribution
        # p_power[n] = highest power of spf[n] dividing n
        # sigma_p[n] = sigma of the p-power part
        
        p_power = [0] * (MAX + 1)
        sigma_p = [0] * (MAX + 1)
        
        for n in range(2, MAX + 1):
            p = spf[n]
            if n % (p * p) == 0:
                # n = p^k * m where k >= 2
                prev = n // p
                p_power[n] = p_power[prev] * p
                sigma_p[n] = sigma_p[prev] + p_power[n]
                m = n // p_power[n]
                sigma[n] = sigma_p[n] * sigma[m]
            else:
                # n = p * m where p ∤ m
                p_power[n] = p
                sigma_p[n] = 1 + p
                m = n // p
                sigma[n] = sigma_p[n] * sigma[m]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [11, 6]
    
    # Now search for best pair
    best_err_num = None
    best_err_den = None
    best_a = 11
    best_b = 6
    
    # a/b ≈ 3/2 means 2a ≈ 3b, i.e., a ≈ 1.5b
    # For each b, check a candidates near 1.5b
    
    for b in range(2, MAX + 1):
        if time.time() > deadline:
            break
        
        sb = sigma[b]
        if sb == 0:
            continue
        
        # Target a ≈ 3b/2
        if b % 2 == 0:
            a_target = (3 * b) // 2
            candidates = [a_target + d for d in range(-5, 6) if d != 0]
        else:
            a_target_lo = (3 * b - 1) // 2
            a_target_hi = (3 * b + 1) // 2
            candidates = []
            for base in (a_target_lo, a_target_hi):
                for d in range(-3, 4):
                    c = base + d
                    if c not in candidates:
                        candidates.append(c)
        
        for a_cand in candidates:
            if 1 < a_cand <= MAX:
                if sigma[a_cand] == sb:
                    num = abs(2 * a_cand - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    if best_err_num is None or num * best_err_den < best_err_num * den:
                        best_err_num = num
                        best_err_den = den
                        best_a = a_cand
                        best_b = b
    
    if best_err_num is not None:
        return [int(best_a), int(best_b)]
    else:
        return [11, 6]
# EVOLVE-BLOCK-END
