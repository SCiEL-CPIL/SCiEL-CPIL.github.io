import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 110  # Safety margin for 120s limit

    # Miller-Rabin for determinism on 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        # Small primes for quick check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        
        # Write n-1 as d*2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    # Precompute small primes for trial division
    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    small_primes_list = generate_primes(1000)

    def has_large_factor(n, p):
        """
        Returns True if n has any prime factor > p.
        Returns False if all prime factors are <= p.
        """
        # Trial division by small primes
        for sp in small_primes_list:
            if sp > p:
                # If the trial division prime exceeds p, we stop.
                # Any remaining factor n is either prime or composite.
                # If n is prime and n > p, it fails.
                # If n is composite, it has factors.
                # But we've removed all factors < sp.
                # Since sp > p, all remaining factors are > p?
                # Not necessarily. n could be 1.
                # If n > 1 and n is not divisible by any prime <= p, 
                # then its smallest prime factor is > p.
                # Thus n has a prime factor > p.
                return n > 1
            
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
                # Check if remaining n is prime
                if is_prime(n):
                    return n > p
                # If n is composite, it might have large factors.
                # We continue to the next iteration or use Pollard.
                # But wait, if we break out of trial division because sp > p is not met yet,
                # we continue.
                # If we finish trial division (all sp <= p) and n > 1:
                # We need to check n.
                # The loop structure above: if sp > p, we return n > 1.
                # This is correct because if n has no factor <= p, its smallest factor is > p.
                # So if n > 1, it has a factor > p.
                
        # If we finished trial division without sp > p, 
        # it means we checked all primes up to 1000 (or up to p if p < 1000).
        # If p >= 1000, we have removed all factors <= 1000.
        # n might still have factors between 1000 and p.
        # Or n might be prime.
        # Or n might be composite with large factors.
        
        if n == 1:
            return False
            
        if is_prime(n):
            return n > p
            
        # n is composite. We need to find a factor.
        # If we find a factor f <= p, we divide and repeat.
        # If we find a factor f > p, we return True.
        # Pollard-Brent finds a factor.
        while n > 1:
            # Optimization: if n is small, trial divide
            # But we already did small primes.
            f = pollard_brent(n)
            if f > p:
                return True
            n //= f
            if n == 1:
                return False
            if is_prime(n):
                return n > p
            # If n is still composite, loop again
            
        return False

    def get_random_prime(min_val, max_val):
        """Generate a random prime in [min_val, max_val]"""
        while True:
            # Ensure odd
            n = random.randint(min_val, max_val)
            if n % 2 == 0:
                n += 1
            if n > max_val:
                continue
            if is_prime(n):
                return n

    best_p = 0
    best_k = 0

    # Phase 1: Sequential scan of small primes
    # Primes up to 10^5 are fast to check.
    primes_small = generate_primes(10**5)
    
    for p in primes_small:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT:
                break
            n = base + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search in larger ranges
    # Allocate remaining time to random sampling
    # Ranges: [10^5, 10^6], [10^6, 10^7], [10^7, 10^8], [10^8, 10^9]
    ranges = [
        (10**5, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9)
    ]
    
    # We will try to sample as many primes as possible
    # Estimate cost:
    # Generating random prime: ~ log(log(N)) MR tests. Fast.
    # Checking k: Each step involves has_large_factor.
    # has_large_factor is the bottleneck.
    # If k is small (e.g. < 5), it's fast.
    # If k is large, it's slow.
    # Most random primes will have small k.
    
    for min_val, max_val in ranges:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Try many attempts
        # Let's try 1000 attempts per range, stopping if time runs out
        for _ in range(1000):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            p = get_random_prime(min_val, max_val)
            k = 0
            base = p * p
            while k < 200:
                if time.time() - start_time > TIME_LIMIT:
                    break
                n = base + k + 1
                if has_large_factor(n, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p

    # Fallback
    if best_p == 0:
        # Should not happen, but just in case
        best_p = 7
        best_k = 1 # p=7, 49+1=50 (2,5) OK, 49+2=51 (3,17) Fail. k=1.

    return {'p': best_p, 'k': best_k}

if __name__ == '__main__':
    construct()
