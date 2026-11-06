import math
import time
import random

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    # Sieve for small primes
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    # Small primes for trial division
    # 10^6 is about 78k primes, might be too many for trial division per number?
    # 10^5 is 9592 primes. 200 numbers * 9592 divisions = ~2M ops. Fast.
    small_primes = sieve_primes(10**5)
    
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
        if n % 5 == 0:
            return 5
        while True:
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def is_p_smooth(n, p):
        if n <= 1:
            return True
        
        # Trial division by small primes
        R = n
        for sp in small_primes:
            if sp > p:
                # If sp > p, and R > 1, then R has a factor >= sp > p
                # So R is not p-smooth
                return R == 1
            if sp * sp > R:
                break
            if R % sp == 0:
                while R % sp == 0:
                    R //= sp
        
        if R == 1:
            return True
            
        if is_prime(R):
            return R <= p
            
        # R is composite and has no prime factors <= 10^5
        # Find a factor
        d = pollard_brent(R)
        if d == R:
             # Should not happen if is_prime is correct, but fallback
             return R <= p
             
        # Check if d is p-smooth
        if d > p:
            return False
        # d <= p. Since d has no small prime factors (from trial division on R, and d divides R),
        # d must be prime or 1? No, d could be composite with large factors.
        # But we know d has no factors <= 10^5.
        # So if d <= p, is it p-smooth?
        # If d is prime, yes.
        # If d is composite, it must have factors > 10^5.
        # Since d <= p, and we are checking if d is p-smooth, we need to recurse.
        
        # Optimization: If d is prime, we are good.
        if is_prime(d):
            if d > p: return False
            # Now check R/d
            return is_p_smooth(R // d, p)
        else:
            # d is composite. Recurse on d and R/d
            if not is_p_smooth(d, p):
                return False
            if not is_p_smooth(R // d, p):
                return False
            return True

    def find_smooth_run(p, max_k=200):
        base = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = base + i
            if not is_p_smooth(n, p):
                break
            k = i
        return k

    def segmented_sieve(low, high):
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        # Reuse small primes if possible
        if limit <= small_primes[-1]:
             small = [p for p in small_primes if p <= limit]
        else:
             small = sieve_primes(limit)
             
        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes

    best_k = 0
    best_p = 2

    # Strategy: Search large primes first, as they have a higher threshold for smoothness.
    # Range: 10^8 to 10^9.
    # We will scan in chunks of 10^6.
    
    # To make it faster, we can skip primes if p^2+1 is not smooth.
    # Most primes will fail at k=1.
    
    # Let's iterate from high to low to find a good k quickly?
    # Or just scan up.
    
    low = 10**8
    high = 10**9
    chunk_size = 10**6
    
    while low < high and time.time() < deadline - 5:
        end = min(low + chunk_size, high)
        primes = segmented_sieve(low, end)
        
        for p in primes:
            if time.time() > deadline - 5:
                break
                
            # Quick check for k=1
            if not is_p_smooth(p*p + 1, p):
                continue
                
            # If k=1 passes, check further
            k = find_smooth_run(p)
            if k > best_k:
                best_k = k
                best_p = p
                
        low = end
        
    return {'p': best_p, 'k': best_k}
