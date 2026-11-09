"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _construct_sieve():
    """Sieve-based enumeration of k*sigma(k) to find M with many solutions."""
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Determine max k we can sieve within time budget
    # k*sigma(k) <= 10^18 and k <= sqrt(M)
    # For small k, sigma(k) ~ O(k log log k), so k*sigma(k) ~ O(k^2 log log k)
    # We need k*sigma(k) <= 10^18, so roughly k <= ~10^9 / log(log(10^9))
    # But we also need k <= sqrt(M) where M = k*sigma(k), so k <= sqrt(k*sigma(k))
    # which means k <= sigma(k), always true for k >= 1.
    # Actually the constraint is k_i <= isqrt(M). Since M = k_i * sigma(k_i),
    # we need k_i <= isqrt(k_i * sigma(k_i)), i.e., k_i^2 <= k_i * sigma(k_i),
    # i.e., k_i <= sigma(k_i). This is always true.
    # So the real constraint is M <= 10^18, i.e., k*sigma(k) <= 10^18.
    
    # Let's sieve up to some max_k. Start with a reasonable bound.
    # sigma(k) for k up to N: use a sieve for the smallest prime factor.
    
    # Strategy: compute sigma(k) for all k up to max_k using a modified sieve.
    # Then compute M = k * sigma(k) and count frequencies.
    
    # Time estimate: sieving up to N takes O(N log log N) time.
    # Let's try N = 5e6 first, then scale up if time permits.
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    for max_k in [5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]:
        if time.time() > deadline - 20:
            break
        
        # Sieve for smallest prime factor (spf)
        spf = list(range(max_k + 1))  # smallest prime factor
        for i in range(2, int(max_k**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i*i, max_k + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Compute sigma(k) for all k using the spf array
        # sigma is multiplicative. sigma(p^a) = (p^(a+1)-1)/(p-1)
        # We can compute it iteratively using spf.
        sigma = [1] * (max_k + 1)
        # sigma[k] = sigma[k // p^a] * (p^(a+1) - 1) / (p - 1) where p = spf[k]
        # Better: use a different approach.
        # Let's use: for each prime p, for each multiple of p, add p^a * (p^(a+1)-1)/(p-1) contribution.
        # Actually, let's use the standard approach:
        # sigma[k] can be computed as:
        # Let p = spf[k], let k = p^a * m where gcd(m, p) = 1
        # sigma[k] = sigma[m] * (p^(a+1) - 1) / (p - 1)
        
        # More efficient: iterate and build up
        # sigma[1] = 1
        # For k >= 2:
        #   p = spf[k]
        #   Let k = p^a * m, p not dividing m
        #   sigma[k] = sigma[m] * (p^(a+1) - 1) // (p - 1)
        
        # We can precompute by iterating:
        # For each k, find p = spf[k], then k = p * (k//p)
        # We need to track the power of p in k.
        
        # Alternative efficient method:
        # Use the fact that sigma is multiplicative and compute via:
        # For each prime p <= max_k, for each power p^a <= max_k,
        #   for each multiple m of p^a where gcd(m, p) = 1 and m*p^a <= max_k:
        #     sigma[m*p^a] += sigma[m] * (p^(a+1)-1)//(p-1) - sigma[m] * (p^a - 1)//(p-1)
        # This is getting complex. Let's use a simpler O(N log N) approach.
        
        # Simple approach: for each d from 1 to max_k, for each multiple m of d up to max_k:
        #   sigma[m] += d
        # This is O(N log N) which for N=100M might be too slow.
        
        # Let's use the spf-based approach properly.
        sigma = [0] * (max_k + 1)
        sigma[1] = 1
        for k in range(2, max_k + 1):
            p = spf[k]
            # Find the highest power of p dividing k
            m = k
            power = 0
            while m % p == 0:
                m //= p
                power += 1
            # k = p^power * m, gcd(m, p) = 1
            # sigma[k] = sigma[m] * (p^(power+1) - 1) // (p - 1)
            # Compute p^(power+1) - 1
            p_pow = 1
            for _ in range(power + 1):
                p_pow *= p
            p_pow_minus_1 = p_pow - 1
            sigma[k] = sigma[m] * p_pow_minus_1 // (p - 1)
        
        if time.time() > deadline - 10:
            break
        
        # Now compute M = k * sigma(k) and find collisions
        M_to_ks = defaultdict(list)
        for k in range(1, max_k + 1):
            M = k * sigma[k]
            if M > 10**18:
                continue
            # Check k <= isqrt(M)
            if k * k <= M:  # k <= sqrt(M)
                M_to_ks[M].append(k)
        
        if time.time() > deadline - 5:
            break
        
        # Find the M with the most k's
        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:1000]
        
        if best_count >= 10:
            break  # Good enough for now
    
    if best_count > 3:
        return {'M': best_M, 'k': best_k}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}

def construct():
    deadline = time.time() + 140
    try:
        result = _construct_sieve()
    except Exception:
        result = {'M': 333312, 'k': [336, 372, 434]}
    return result
# EVOLVE-BLOCK-END
