import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Small primes for trial division
    # Generate primes up to 1000
    def sieve(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = 0
        is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i, val in enumerate(is_p) if val]
    
    small_primes = sieve(1000)
    
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in _MR_BASES:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n - 1)
            c = random.randrange(1, n - 1)
            m = 128
            g = r = q = 1
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
            if g != n:
                return g

    def is_p_smooth(n, p):
        """Check if n is p-smooth (all prime factors <= p)."""
        if n == 1:
            return True
        
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
        
        if n == 1:
            return True
            
        # If n is prime
        if is_prime(n):
            return n <= p
            
        # If n is composite
        # Find a factor
        d = pollard_brent(n)
        if d == n:
            # Should not happen if is_prime worked, but safety
            return n <= p
            
        # Recursively check factors
        # We can short-circuit if one fails
        if not is_p_smooth(d, p):
            return False
        if not is_p_smooth(n // d, p):
            return False
        return True

    def run_length(p, max_k=200):
        p_sq = p * p
        for i in range(1, max_k + 1):
            n = p_sq + i
            if not is_p_smooth(n, p):
                return i - 1
        return max_k

    # Generate candidate primes
    # We want to check as many primes as possible.
    # Sieve up to 2 million first.
    limit1 = 2000000
    primes = sieve(limit1)
    
    best_k = 0
    best_p = 2
    
    # Check primes
    for p in primes:
        if time.time() > deadline:
            break
        
        # Quick check: p^2+1
        # If p^2+1 is prime and > p, it fails immediately.
        # is_p_smooth handles this, but checking is_prime(p^2+1) might be faster?
        # No, is_p_smooth does trial division first.
        
        k = run_length(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            
    # If time remains, check larger primes
    # Use random sampling to avoid sieving huge ranges
    if time.time() < deadline - 5:
        random.seed(42)
        # Try random primes in [2e6, 1e9]
        # We need to find primes.
        # Generate random odd numbers and check primality.
        
        # To be efficient, we can just try random numbers and if prime, check run.
        # But finding a prime takes ~ln(n) trials.
        # Maybe sieve a smaller segment?
        
        # Let's just do random search
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Random candidate
            # Bias towards larger numbers? No, probability is similar.
            # But larger p means larger n, which might be slower to factor.
            # So smaller p is better for speed.
            # We already checked up to 2e6.
            # Let's check primes between 2e6 and 5e6.
            
            candidate = random.randint(2000000, 5000000)
            # Ensure odd
            if candidate % 2 == 0:
                candidate += 1
                
            # Find next prime
            # Simple loop
            found_prime = False
            for _ in range(100): # Max 100 tries
                if is_prime(candidate):
                    found_prime = True
                    break
                candidate += 2
                
            if not found_prime:
                continue
                
            if candidate > 10**9:
                continue
                
            k = run_length(candidate, 200)
            if k > best_k:
                best_k = k
                best_p = candidate

    return {'p': best_p, 'k': best_k}
