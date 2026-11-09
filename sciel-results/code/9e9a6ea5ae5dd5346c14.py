import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MILLER_RABIN_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MILLER_RABIN_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
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
                if g != 1:
                    break
        if g != n:
            return g

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _segmented_sieve(start, end):
    """Yield primes in [start, end]"""
    if end < 2:
        return
    limit = int(math.isqrt(end)) + 1
    small_primes = _sieve(limit)
    
    # Create a bytearray for the segment
    size = end - start + 1
    is_prime = bytearray(b'\x01') * size
    
    for p in small_primes:
        if p * p > end:
            break
        # Find the first multiple of p in [start, end]
        first = ((start + p - 1) // p) * p
        if first == p:
            first += p
        for j in range(first - start, size, p):
            is_prime[j] = 0
            
    for i in range(size):
        if is_prime[i]:
            yield start + i

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute small primes for trial division
    SMALL_PRIME_LIMIT = 1000000
    small_primes = _sieve(SMALL_PRIME_LIMIT)
    
    best_k = 0
    best_p = 3
    
    # Search primes in descending order from 10^9 to 10^6
    # We use segments of size 10^6 to keep memory low
    SEGMENT_SIZE = 1000000
    current_end = 10**9
    current_start = max(10**6, current_end - SEGMENT_SIZE + 1)
    
    while current_start <= 10**9 and time.time() < deadline:
        if current_end < 10**6:
            break
            
        # Generate primes in [current_start, current_end]
        # Note: _segmented_sieve yields in ascending order.
        # We want descending, so we collect and reverse, or just iterate and check.
        # For efficiency, let's collect them in a list and reverse.
        # Segment size 10^6 has ~78k primes. List is small.
        primes_in_segment = list(_segmented_sieve(current_start, current_end))
        
        # Iterate in reverse (descending)
        for p in reversed(primes_in_segment):
            if time.time() > deadline:
                break
            if p > 10**9:
                continue
            if p < 10**6:
                continue
                
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                t = val
                has_large_factor = False
                
                # Trial division by small primes
                for q in small_primes:
                    if q > p:
                        break
                    if q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                
                if t == 1:
                    k = i
                    continue
                
                # t > 1
                if t <= p:
                    k = i
                else:
                    # t > p
                    # Check if t is prime
                    if _is_prime_mr(t):
                        has_large_factor = True
                    else:
                        # t is composite, factor it
                        factors = []
                        _factor(t, factors)
                        if all(f <= p for f in factors):
                            k = i
                        else:
                            has_large_factor = True
                
                if has_large_factor:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break
        
        # Move to next segment
        current_end = current_start - 1
        current_start = max(10**6, current_end - SEGMENT_SIZE + 1)
        
        if best_k == 200:
            break

    # If we didn't find 200, try some random large primes just in case
    if best_k < 200:
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Sample primes in [10^8, 10^9]
            p = random.randint(10**8, 10**9)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
            if p > 10**9:
                continue
            
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                t = val
                has_large_factor = False
                
                for q in small_primes:
                    if q > p:
                        break
                    if q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                
                if t == 1:
                    k = i
                    continue
                
                if t <= p:
                    k = i
                else:
                    if _is_prime_mr(t):
                        has_large_factor = True
                    else:
                        factors = []
                        _factor(t, factors)
                        if all(f <= p for f in factors):
                            k = i
                        else:
                            has_large_factor = True
                
                if has_large_factor:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break

    return {'p': best_p, 'k': best_k}
