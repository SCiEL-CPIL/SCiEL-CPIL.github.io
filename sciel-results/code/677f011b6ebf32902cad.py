import math
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 130s usage
    DEADLINE = start_time + 130.0
    
    # 1. Precompute primes up to 10^6
    # sqrt(10^12) is 10^6. If we factor n, we need primes up to sqrt(remaining).
    # If we only go up to 10^6, the remaining factor is either 1 or prime (if n <= 10^12).
    # Actually, if n = p*q with p,q > 10^6, then n > 10^12. 
    # So any composite factor > 10^6 cannot exist for n <= 10^12.
    # Thus, after removing all prime factors <= 10^6, the remainder is 1 or a prime.
    
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Miller-Rabin for primality testing of large numbers
    def is_prime_mr(n):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0: return False
        
        # Deterministic bases for n < 3.3 * 10^24
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Optimized phi calculation
    def phi(x):
        if x == 1: return 1
        n = x
        result = x
        # Trial division by small primes
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            # n is either 1 or a prime > 10^6
            # Since x <= 10^12, n can be at most 10^12.
            # If n > 1, it must be prime because any composite factor would have a prime factor <= sqrt(n) <= 10^6.
            # Wait, if n is composite, it has a prime factor <= sqrt(n).
            # If sqrt(n) > 10^6, then n > 10^12, which is impossible.
            # So n is definitely prime if n > 1.
            result -= result // n
        return result

    best_k = 1
    best_s = 1
    
    # 4. Search Strategy
    # We want to find a start s such that phi(s), ..., phi(s+k-1) are distinct.
    # Random sampling of s in [1, 10^12].
    # For each s, we compute phi(s), phi(s+1), ... until a collision or limit k=200 or time out.
    
    # To speed up, we can check collisions quickly.
    # We'll sample random ranges.
    # Also, check small ranges specifically? No, random is better for global max.
    
    # Let's do a mix:
    # 1. Check some small ranges to ensure we have a valid baseline (though seed did this).
    # 2. Random sampling.
    
    # Helper to check a specific start
    def check_start(s):
        seen = set()
        k = 0
        limit_k = 200
        while k < limit_k:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    # Phase 1: Quick check of small numbers to establish a baseline if random fails early
    # (Though seed found k=1..something, let's just rely on random for improvement)
    
    # Phase 2: Random Search
    # We need to be careful with time.
    # phi() takes ~microseconds for small primes, but more for large.
    # Let's estimate. 10^6 primes. Loop breaks when p*p > n.
    # For n ~ 10^12, we iterate primes up to 10^6. There are ~78,000 primes.
    # This is too slow per call if we do many.
    # Optimization: We only need to factor n.
    # If we just do trial division up to sqrt(n), it's slow.
    # Using the list of primes is good.
    
    # Let's refine phi to be faster.
    # If we just want to check distinctness, do we need exact phi? Yes.
    
    # Is there a faster way?
    # Maybe precompute phi for small ranges? No, range is 10^12.
    
    # Let's optimize the inner loop of phi.
    # Most numbers are even.
    
    # Let's try a few random samples to gauge speed.
    
    # Random starts
    # We will sample random s in [1, 10^12 - 200]
    
    # To avoid duplicates and cover space, we can use random.
    
    # Time check inside loop
    while time.time() < DEADLINE:
        # Choose a random start
        # Bias towards larger numbers? Or uniform?
        # Uniform is safer.
        s = random.randint(1, 10**12 - 200)
        
        # Check this start
        k = check_start(s)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        # If we found a very long run, we might stop or continue?
        # Max k is 200. If we hit 200, we can stop.
        if best_k == 200:
            break
        
        # If we are running out of time, break
        if time.time() > DEADLINE - 5.0:
            break

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
