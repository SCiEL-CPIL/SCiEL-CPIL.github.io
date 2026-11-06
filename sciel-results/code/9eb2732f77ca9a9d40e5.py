import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s buffer for safety
    
    # Constants for Miller-Rabin
    # Deterministic for 64-bit integers
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = random.randrange(1, n)
            g, q, r = 1, 1, 1
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

    def factorize(n, limit):
        """
        Returns True if all prime factors of n are <= limit.
        Uses trial division for small primes, then Pollard Rho.
        """
        if n <= 1:
            return True
        
        # Trial division by small primes
        # We can hardcode a few small primes to speed up
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                if p > limit:
                    return False
                while n % p == 0:
                    n //= p
                if n == 1:
                    return True
                # Check if remaining n is prime and <= limit
                if is_prime(n):
                    return n <= limit
                # Otherwise, continue to Pollard Rho on n
                break
        
        # If n is prime
        if is_prime(n):
            return n <= limit
        
        # Pollard Rho
        # Find a factor
        d = pollard_brent(n)
        # Recurse on factors
        return factorize(d, limit) and factorize(n // d, limit)

    def check_run(p):
        """
        Returns k such that p^2+1 ... p^2+k are p-smooth.
        """
        k = 0
        base = p * p
        # We can check up to a reasonable max k, say 1000
        # But the task says maximize k, and k<=200 is the constraint for validity,
        # but we want to maximize it. Let's check up to 1000 to be safe, 
        # but stop if we exceed time or if k becomes very large (unlikely).
        while k < 1000:
            n = base + k + 1
            if not factorize(n, p):
                break
            k += 1
        return k

    # Sieve of Eratosthenes to get primes
    # We will generate primes in chunks or up to a limit.
    # Let's start with primes up to 10^6, then extend if time permits.
    # Actually, generating primes up to 10^7 takes ~1s and 10MB.
    # Let's use a simple sieve up to 10^6 first.
    
    limit = 10**6
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2
    
    # Iterate through primes
    # We can shuffle or just go sequential. Sequential is deterministic.
    # However, large p might have larger k.
    # Let's iterate sequentially but check time often.
    
    # Optimization: If p is very small, k is likely small.
    # Let's just iterate.
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Optional: Skip very small primes if we already found a good k?
        # No, we want max k.
        
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            
        # If we found a very large k, we might want to stop?
        # No, keep looking.
        
        # If time is running out, break
        if time.time() > deadline:
            break

    # If we have time left, we can try larger primes using Miller-Rabin
    # to generate candidates.
    # But checking a random large prime is expensive if k is small.
    # However, if we haven't found k > 10, we might want to try larger primes.
    # For now, the sieve up to 10^6 covers many cases.
    
    return {'p': best_p, 'k': best_k}
