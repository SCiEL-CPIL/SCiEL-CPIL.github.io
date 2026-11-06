import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin below 150s limit

    # 1. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # 2. Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
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
            if g != n:
                return g

    # 3. Generate primes up to 10^7 using a simple sieve
    # This provides both candidate primes p and a list for trial division.
    limit = 10**7
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime_flag in enumerate(sieve) if is_prime_flag]

    # 4. Helper: Check if n is p-smooth
    # Optimized: Trial divide by small primes first, then use Pollard-Brent for the remainder.
    # This avoids iterating over ~600k primes for every number.
    def is_p_smooth(n, p, small_primes):
        if n <= 1:
            return True
        if n <= p:
            return True # All factors of n are <= n <= p
        
        # Divide by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if n <= p:
                    return True
                # If n is now prime, check if <= p
                if is_prime(n):
                    return n <= p
                # If n is composite and > p, we might need more factors
                # But if we've divided by small primes, n is likely a product of larger primes
                # or a prime > p.
                # We break out to handle the rest with Pollard-Brent or recursion if needed.
                # Actually, if n > p and we haven't reduced it to 1 or a prime <= p, 
                # it might still be p-smooth if it has factors <= p that are larger than our small_primes list.
                # So we cannot just return False here. We must factor the remainder.
                break
        
        if n == 1:
            return True
        if n <= p:
            return True
            
        # Check if remainder is prime
        if is_prime(n):
            return n <= p
        
        # Use Pollard-Brent to find a factor
        # If we find a factor f:
        # If f > p, return False (since f is a prime factor > p)
        # If f <= p, recurse on f and n//f
        # Note: Pollard-Brent finds a non-trivial factor. It might not be prime.
        # We need to ensure all prime factors are <= p.
        
        def check_factor(m, p):
            if m == 1:
                return True
            if m <= p:
                return True
            if is_prime(m):
                return m <= p
            
            # Try to find a factor
            f = None
            for _ in range(10): # Limit attempts
                f = pollard_brent(m)
                if f is not None and 1 < f < m:
                    break
            if f is None or f == m:
                # Fallback: assume prime? Or timeout?
                # If we can't factor it, it's risky. 
                # Given n ~ p^2, and we've stripped small factors, 
                # if it's not prime, it's likely composite with large factors.
                # If m > p and we can't factor, it's safer to assume it has a large factor?
                # Actually, if m is composite and > p, it could be p-smooth (e.g., p*p).
                # But we stripped small primes.
                # Let's try harder or just return False to be safe?
                # No, if it's p-smooth, it's valid. 
                # But if we can't factor, we don't know.
                # However, for n ~ 10^14, Pollard-Brent is very fast.
                # If it fails 10 times, something is wrong or it's prime (but is_prime said no).
                # Let's just return False to avoid false positives (claiming smooth when not).
                return False
            
            return check_factor(f, p) and check_factor(m // f, p)
            
        return check_factor(n, p)

    # Precompute small primes for trial division (up to 10,000 is usually enough to strip most factors)
    # We can just use the first part of the 'primes' list
    small_primes = [q for q in primes if q <= 10000]

    # Main search loop
    best_k = 0
    best_p = 0
    
    # We iterate through primes p.
    # Larger p generally allows for longer runs?
    # Actually, the condition is "all prime factors <= p".
    # For a fixed i, p^2 + i.
    # If p is small, p^2 + i is small, factors are small.
    # If p is large, p^2 + i is large.
    # It's not obvious which p gives the longest run.
    # We should search broadly.
    
    # To maximize k, we want to find a p where p^2+1 ... p^2+k are all p-smooth.
    # This is a rare event.
    # We'll scan primes.
    
    # Optimization: 
    # If we find a run of length k, we can stop early for that p.
    # We want to maximize k globally.
    
    # Let's scan primes in increasing order? Or random?
    # Increasing order is deterministic.
    # Let's try scanning from small to large, but skip very small ones (k is likely small).
    # Actually, small p means p^2+i is small. 
    # Example p=3: 9+1=10 (2*5, 5>3 fail). k=0.
    # p=5: 25+1=26 (2*13, 13>5 fail). k=0.
    # p=7: 49+1=50 (2*5*5, 5<=7 ok). 49+2=51 (3*17, 17>7 fail). k=1.
    # p=11: 121+1=122 (2*61, 61>11 fail).
    
    # It seems small p often fail immediately.
    # Larger p might have better chances.
    
    # Let's iterate through all primes up to 10^7.
    # This is ~660,000 primes.
    # For each, we check up to 200 numbers.
    # If the first number fails, it's very fast.
    # If it succeeds, it's slower.
    
    # We need to be careful with time.
    # 660k primes * (time per check).
    # If average check is 0.1 ms, 660k * 0.1ms = 66s.
    # This might be tight.
    
    # We can skip primes that are "bad" quickly.
    # Check p^2+1 first. If it fails, move on.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        if p < 10:
            continue
            
        # Quick check for p^2+1
        n1 = p * p + 1
        if not is_p_smooth(n1, p, small_primes):
            continue
            
        # If p^2+1 is smooth, check further
        k = 1
        for i in range(2, 201):
            if time.time() > deadline:
                break
            n = p * p + i
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
