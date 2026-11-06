"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def phi_fast(n, primes):
    """Compute phi(n) using precomputed primes."""
    result = n
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            result -= result // p
            while temp % p == 0:
                temp //= p
    if temp > 1:
        result -= result // temp
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate primes up to 10^6 for factoring numbers up to 10^12
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    def check_run(s, k, primes):
        """Check if phi(s), ..., phi(s+k-1) are all distinct. Returns (valid, collision_index)."""
        seen = set()
        for i in range(k):
            n = s + i
            v = phi_fast(n, primes)
            if v in seen:
                return False, i
            seen.add(v)
        return True, None
    
    # Strategy: Use sliding window approach with segmented sieve for efficiency
    # First, try to find a run of length 200 by scanning candidate regions
    
    best_result = {'start': 1, 'length': 1}
    
    # Generate candidate starting positions - focus on regions where collisions are less likely
    # Large numbers have more "random" totient behavior
    
    candidates = []
    
    # Near the top boundary (most likely to work)
    base = 10**12 - 200
    for offset in range(0, 10000, 25):
        s = base - offset
        if s >= 1:
            candidates.append(s)
    
    # Spread across the range with denser sampling near high values
    for frac in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
        s = int(10**12 * frac)
        for delta in range(-2000, 2001, 50):
            s2 = s + delta
            if 1 <= s2 <= 10**12 - 200:
                candidates.append(s2)
    
    # Pseudo-random candidates
    rng = random.Random(12345)
    for _ in range(200):
        s = rng.randint(10**11, 10**12 - 200)
        candidates.append(s)
    
    candidates = list(set(candidates))
    candidates.sort(reverse=True)
    
    # For each candidate, try k=200 first
    for s in candidates:
        if time.time() > deadline - 15:
            break
        
        if s + 199 > 10**12:
            continue
        
        valid, _ = check_run(s, 200, primes)
        if valid:
            return {'start': s, 'length': 200}
    
    # If k=200 wasn't found, try progressively smaller k
    for k in range(199, 50, -1):
        if time.time() > deadline - 10:
            break
        
        for s in candidates:
            if time.time() > deadline - 10:
                break
            if s + k - 1 > 10**12 or s < 1:
                continue
            
            valid, _ = check_run(s, k, primes)
            if valid:
                if k > best_result['length']:
                    best_result = {'start': s, 'length': k}
                return best_result
    
    # Final fallback: try a few more random positions with smaller k
    rng2 = random.Random(99999)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        s = rng2.randint(1, 10**12 - 200)
        for k in [100, 80, 60, 40, 20]:
            if s + k - 1 > 10**12:
                continue
            valid, _ = check_run(s, k, primes)
            if valid:
                if k > best_result['length']:
                    best_result = {'start': s, 'length': k}
                return best_result
    
    return best_result

# EVOLVE-BLOCK-END
