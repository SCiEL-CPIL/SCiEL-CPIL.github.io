import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Determine N based on available time
    # Sieve is O(N log N). Grouping is O(N).
    # Let's try N = 50,000,000 first. If it's fast, we can't easily increase it dynamically 
    # without restarting the sieve. 
    # However, we can just pick a large N.
    N = 50000000
    
    # 1. Sieve sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 30:
        # Emergency: return best found so far from a smaller subset? 
        # Actually, if sieve is done, grouping takes time.
        # If we are here, sieve is done.
        pass

    # 2. Group k by M
    # This loop is the bottleneck.
    # Using a list of lists or defaultdict.
    m_to_k = defaultdict(list)
    # Convert to list for faster access? No, numpy indexing is fast.
    # But iterating in Python is slow.
    # We can try to vectorize the M calculation, but storing in dict is hard.
    # Let's stick to Python loop but optimize it.
    
    # Pre-compute M values? No, memory.
    # Just loop.
    # To speed up, we can skip k where M > 10^18.
    # k * sigma(k) <= 10^18 => sigma(k) <= 10^18 / k.
    
    max_M = 10**18
    for k in range(1, N + 1):
        sk = sigma[k]
        # Quick check to avoid overflow or large values
        if sk > max_M // k:
            continue
        M_val = k * sk
        m_to_k[M_val].append(k)
        
    if time.time() > deadline - 20:
        # Find best from m_to_k quickly
        best_M, best_ks = _find_best_from_map(m_to_k)
        return _finalize(best_M, best_ks)

    # 3. Find best M
    best_count = 0
    best_M = 0
    best_ks = []
    
    for M_val, ks in m_to_k.items():
        # Filter k <= sqrt(M)
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M_val
            best_ks = valid_ks

    if best_M == 0:
        # Fallback
        return {'M': 1, 'k': [1]}

    # 4. Extension Search
    # Factorize best_M
    # Generate all divisors k <= sqrt(best_M)
    # Check if k*sigma(k) == best_M
    
    divisors = _get_divisors(best_M)
    sqrt_M = math.isqrt(best_M)
    
    # We already have valid_ks from the sieve.
    # We need to check divisors that were NOT in the sieve (k > N) or 
    # verify all divisors to be sure.
    # Actually, the sieve found all k <= N.
    # So we only need to check divisors k > N.
    # But we must ensure k <= sqrt_M.
    
    final_ks = set(best_ks)
    
    # Pre-compute primes for factorization of k
    # Primes up to sqrt(sqrt(M)) ~ 31622 for M=10^18? No.
    # k <= sqrt(M) <= 10^9. sqrt(k) <= 31622.
    # So primes up to 40000 are sufficient for factoring k.
    
    primes = _get_primes(40000)
    
    for k in divisors:
        if k > N and k <= sqrt_M:
            # Compute sigma(k)
            sk = _sigma_by_factorization(k, primes)
            if k * sk == best_M:
                final_ks.add(k)
                
        # Time check
        if time.time() > deadline - 5:
            break

    # Sort final_ks
    final_ks_list = sorted(final_ks)
    
    return _finalize(best_M, final_ks_list)

def _find_best_from_map(m_to_k):
    best_count = 0
    best_M = 1
    best_ks = [1]
    for M_val, ks in m_to_k.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M_val
            best_ks = valid_ks
    return best_M, best_ks

def _finalize(M, ks):
    M = int(M)
    ks = sorted(set(int(k) for k in ks))
    sqrt_M = math.isqrt(M)
    ks = [k for k in ks if k <= sqrt_M]
    return {'M': M, 'k': ks}

def _get_primes(limit):
    """Sieve of Eratosthenes for primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _get_divisors(n):
    """Get all divisors of n."""
    if n <= 0:
        return []
    factors = _factorize(n)
    divisors = [1]
    for p, e in factors:
        current = list(divisors)
        p_power = 1
        for i in range(e):
            p_power *= p
            for d in current:
                divisors.append(d * p_power)
    divisors.sort()
    return divisors

def _factorize(n):
    """Factorize n. Uses trial division with precomputed primes if possible, else wheel."""
    if n <= 1:
        return []
    
    factors = []
    temp = n
    
    # Try small primes first
    # 2, 3, 5
    for p in [2, 3, 5]:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
    
    # Wheel for remaining
    d = 7
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            factors.append((d, count))
        d += 2 # Only check odds, simple optimization
        # A better wheel (2,3,5) would be faster, but this is okay for one number
        
    if temp > 1:
        factors.append((temp, 1))
        
    return factors

def _sigma_by_factorization(n, primes):
    """Compute sigma(n) using precomputed primes."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    temp = n
    result = 1
    
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            # Divide out p
            p_power_sum = 1
            p_power = 1
            while temp % p == 0:
                temp //= p
                p_power *= p
                p_power_sum += p_power
            result *= p_power_sum
            
    if temp > 1:
        # temp is prime
        result *= (1 + temp)
        
    return result
