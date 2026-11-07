"""Rewritten program: search for M with many k such that k*sigma(k)=M."""
import math
import time
import sys
from math import isqrt, prod

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n via trial division."""
    if n == 1:
        return 1
    result = 1
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            sp = 1
            pk = 1
            while m % p == 0:
                m //= p
                pk *= p
                sp += pk
            result *= sp
        p += 1 if p == 2 else 2
    if m > 1:
        result *= (1 + m)
    return result

def _sigma_fast(n, primes):
    """Compute sigma(n) using a list of primes for trial division."""
    if n == 1:
        return 1
    result = 1
    m = n
    for p in primes:
        if p * p > m:
            break
        if m % p == 0:
            sp = 1
            pk = 1
            while m % p == 0:
                m //= p
                pk *= p
                sp += pk
            result *= sp
    if m > 1:
        result *= (1 + m)
    return result

def _gen_primes(limit):
    """Generate primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Precompute primes up to a reasonable bound for fast sigma
    primes = _gen_primes(100000)
    
    # Known good answer as fallback
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy: For each k, compute M = k*sigma(k). Group by M.
    # We want M with many distinct k values.
    # k must be <= sqrt(M) = sqrt(k*sigma(k)), so k^2 <= k*sigma(k), i.e., k <= sigma(k).
    # This is always true for k >= 1 (sigma(k) >= k+1 for k>1, sigma(1)=1).
    # Actually sigma(k) >= k+1 for k>1, so k < sigma(k) for k>1. For k=1, sigma(1)=1, M=1.
    
    # The constraint is k_i <= isqrt(M). Since M = k_i * sigma(k_i),
    # we need k_i <= isqrt(k_i * sigma(k_i)), i.e., k_i^2 <= k_i * sigma(k_i), i.e., k_i <= sigma(k_i).
    # For k_i >= 2, sigma(k_i) >= k_i + 1 > k_i, so this always holds.
    # For k_i = 1, sigma(1) = 1, M = 1, isqrt(1) = 1, so k=1 works.
    
    # So the constraint is automatically satisfied for all k >= 1.
    
    # We want to find M with many preimages k.
    # Let's search k from 1 up to some bound, compute M = k*sigma(k), and count occurrences.
    
    # But M must be <= 10^18. So k*sigma(k) <= 10^18.
    # sigma(k) can be as large as ~k * log(log(k)) for smooth numbers, so k can be up to ~10^18 / log(10^18) ~ 10^18/41 ~ 2.4*10^16.
    # That's too large to iterate.
    
    # Alternative: iterate over M values? No, M space is huge.
    
    # Better approach: For a fixed M, find all k such that k*sigma(k) = M.
    # k must divide M, and sigma(k) = M/k.
    # So k | M and sigma(k) = M/k.
    
    # Strategy: Pick M, factor it, enumerate divisors k of M, check if sigma(k) == M/k.
    
    # To maximize count, we want M with many divisors k where sigma(k) = M/k.
    
    # Let's try a different approach: generate candidate k values, compute M = k*sigma(k),
    # and use a dictionary to count how many k map to each M.
    # We need to be smart about the range of k.
    
    # For small k, M = k*sigma(k) is small. Let's go up to k where M stays <= 10^18.
    # sigma(k) for k up to 10^6 is at most around 10^6 * 4 = 4*10^6, so M up to 4*10^12.
    # For k up to 10^8, sigma(k) ~ 10^8 * 4, M ~ 4*10^16, still under 10^18.
    # For k up to 10^9, sigma(k) ~ 10^9 * 4, M ~ 4*10^18, might exceed.
    
    # Let's iterate k from 1 to some max_k, computing M = k*sigma(k), and tally.
    # Use the prime list for fast sigma computation.
    
    max_k = 5000000  # 5 million, should be fast enough
    counts = {}
    
    for k in range(1, max_k + 1):
        if time.time() > deadline - 10.0:
            break
        s = _sigma_fast(k, primes)
        M = k * s
        if M > 10**18:
            # sigma is non-decreasing in a rough sense, but not strictly. 
            # However, for large k, M will stay > 10^18 mostly.
            # We can't break immediately because sigma isn't monotonic.
            # But for practical purposes, once k is large enough, M will exceed 10^18.
            # Let's just skip.
            continue
        if M in counts:
            counts[M].append(k)
        else:
            counts[M] = [k]
    
    # Find the M with the most k values
    for M, ks in counts.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
    
    # Also check some known constructions
    # The seed has 3 solutions. Let's see if we can do better with targeted search.
    
    # Try to find M with more solutions by looking at the structure.
    # If k1*sigma(k1) = k2*sigma(k2) = M, then k1/k2 = sigma(k2)/sigma(k1).
    
    # Let's also try a targeted search: pick M and enumerate its divisors.
    # For M up to 10^18, we can factor M and enumerate divisors if M has a manageable number of divisors.
    
    # Try some specific M values that might have many solutions.
    # For example, M = product of small primes raised to powers.
    
    # Let's try M values that are highly composite.
    # Actually, let's just go with the dictionary approach result.
    
    # Verify the best result
    if best_k:
        # Sort and deduplicate
        best_k = sorted(set(best_k))
        # Verify each k satisfies k*sigma(k) = M and k <= isqrt(M)
        verified = []
        for k in best_k:
            s = _sigma_fast(k, primes)
            if k * s == best_M and k <= isqrt(best_M):
                verified.append(k)
        if len(verified) > best_count:
            best_count = len(verified)
            best_k = verified
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}
