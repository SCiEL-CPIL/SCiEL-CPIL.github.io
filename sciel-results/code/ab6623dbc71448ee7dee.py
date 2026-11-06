import math
import random
import time

def is_prime(n):
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    if n == 3:
        return True
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Deterministic bases for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    """Pollard's Brent's cycle finding algorithm for factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(2, n)
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

def factorize(n):
    """Fully factor n into a multiset of prime factors."""
    if n == 1:
        return []
    factors = []
    # We don't have access to the global small_primes here easily without passing it,
    # so we'll do a small trial division here as well, or rely on the caller.
    # For the 'new approach', we will do trial division in the main loop.
    # This function is a fallback for the remaining cofactor.
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
        else:
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
    return factors

def check_p_smooth(n, p, small_primes):
    """
    Check if n is p-smooth (all prime factors <= p).
    Uses small_primes for trial division.
    """
    if n == 1:
        return True
    orig_n = n
    # Trial division by small primes
    # We only need to divide by primes <= p
    for q in small_primes:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    if n == 1:
        return True
    # If n is prime
    if is_prime(n):
        return n <= p
    # If n is composite, it has a factor > p?
    # Not necessarily. It could be a product of two primes both <= p.
    # But if we've divided out all primes <= sqrt(n) (which is < p if n < p^2), 
    # wait, n <= p^2 + k. sqrt(n) ~ p.
    # If we divided by all primes <= p, and n > 1, then n must have a prime factor > p.
    # Why? If all prime factors were <= p, they would have been divided out.
    # So if n > 1 after dividing by all primes <= p, n is NOT p-smooth.
    # However, we only iterate small_primes. If p is large, small_primes might not cover up to p.
    # So we need to be careful.
    
    # Revised logic:
    # We divide by primes in small_primes that are <= p.
    # If small_primes covers up to max_small, and p > max_small, we haven't divided by all primes <= p.
    # In that case, the remaining n could be p-smooth.
    # We need to check if the remaining n has any prime factor > p.
    # This is equivalent to checking if n is 1, or if n is prime and <= p, or if n is composite and all its factors are <= p.
    # If n is composite, and we haven't fully factored it, we can't be sure.
    # So we must factor the remainder if it's not prime.
    
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    
    # n is composite. We need to check if it has a factor > p.
    # We can try to factor it.
    factors = factorize(n)
    return all(f <= p for f in factors)

def construct():
    best_k = 0
    best_p = 2
    
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Generate primes up to 10^7
    limit = 10000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i : limit + 1 : i] = bytearray(len(sieve[i*i : limit + 1 : i]))
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # To speed up trial division in check_p_smooth, we might want to limit the list
    # but we need primes up to p. Since p <= 10^9, and our sieve is 10^7,
    # for p > 10^7, we will have to factor the remainder.
    
    for p in small_primes:
        if time.time() - start_time > time_limit:
            break
        
        # Quick check for p^2 + 1
        n = p * p + 1
        if not check_p_smooth(n, p, small_primes):
            continue
        
        k = 1
        while k < 200:
            n = p * p + k + 1
            if check_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
            
    return {'p': best_p, 'k': best_k}
