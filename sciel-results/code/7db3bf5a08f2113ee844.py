import math
import time
import random
import sys

# Deterministic Miller-Rabin bases for n < 2^64
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# Pre-computed small primes for trial division
_SMALL_PRIMES = []
def _init_small_primes(limit=10000):
    if _SMALL_PRIMES:
        return
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    _SMALL_PRIMES.extend([i for i, prime in enumerate(is_prime_arr) if prime])
_init_small_primes(10000)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
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
    if n % 5 == 0:
        return 5
    if n % 7 == 0:
        return 7
    if n % 11 == 0:
        return 11
    
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
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

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes
    for sp in _SMALL_PRIMES:
        if sp > p:
            break
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
            if n == 1:
                return True
    
    if n == 1:
        return True
    
    if is_prime(n):
        return n <= p
    
    d = pollard_brent(n)
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def sieve(limit):
    if limit < 2:
        return []
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def tonelli_shanks(n, p):
    """
    Find a square root of n modulo p (p is prime).
    Returns None if n is not a quadratic residue.
    """
    if n == 0:
        return 0
    if p == 2:
        return n
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    
    # Case p % 4 == 3
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    # Case p % 8 == 5
    if p % 8 == 5:
        v = pow(n, (p - 1) // 4, p)
        if v == 1:
            return pow(n, (p + 3) // 8, p)
        else:
            return (2 * n * pow(4 * n * n, (p - 5) // 8, p)) % p
    
    # Tonelli-Shanks for general p
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while t != 1:
        i = 0
        t2 = t
        while t2 != 1:
            t2 = (t2 * t2) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_p = 2
    best_k = 1
    
    # Phase 1: Sieve primes up to 10^6 for the pre-filtering
    sieve_limit = 10**6
    primes = sieve(sieve_limit)
    
    # Pre-filtering array: bad_j[p] = smallest j such that p^2+j has a prime factor q > p (with q <= sieve_limit)
    # If bad_j[p] > 200, it means no such factor was found for j in [1, 200]
    # We use 0-indexed for p, so bad_j is a list of size sieve_limit+1
    bad_j = [201] * (sieve_limit + 1)
    
    # We need to mark p as bad if there exists q > p (prime) and j in [1, 200] such that q | (p^2+j)
    # This implies p^2 = -j (mod q)
    # So p is a square root of -j mod q.
    # We iterate over primes q. For each q, we check all j in [1, 200].
    # If -j is a QR mod q, find roots r1, r2.
    # If r is prime and r < q, then r is a candidate p that fails at j.
    # We update bad_j[r] = min(bad_j[r], j).
    
    # Optimization: Only q up to some limit? 
    # If q is very large, r < q means r can be up to q. 
    # But we only care about p up to sieve_limit.
    # So we only care about q > p, which means q can be up to sieve_limit (since p < sieve_limit).
    # Actually, if p < sieve_limit, q can be any prime > p.
    # But if q > sieve_limit, we don't have it in our `primes` list.
    # However, if p is small, p^2+j is small, so its factors are small.
    # If p is large (close to 10^6), p^2 is 10^12. Factors can be up to 10^6.
    # So using primes q up to 10^6 covers most cases for p up to 10^6.
    # For p > 10^6, we might miss large factors q > 10^6.
    # But we will explicitly verify candidates later.
    
    # To make the sieve fast, we can limit the j loop.
    # We only care if bad_j[p] > current_best_k.
    # Initially best_k is small.
    
    # Let's run the sieve for all j in 1..200
    # This might be slow if we do it for all q.
    # Number of primes up to 10^6 is ~78000.
    # For each q, 200 checks. 78000 * 200 = 15.6 million checks.
    # Each check involves a Tonelli-Shanks or simple pow.
    # This might be too slow in Python.
    
    # Alternative: Iterate p, then check j?
    # For each p, check j=1..200. If p^2+j has a factor q>p, mark bad.
    # This is what the original program did, but without Pollard-Brent for every j.
    # We can just trial divide p^2+j by primes q > p up to sqrt(p^2+j).
    # sqrt(p^2+j) ~ p. So we trial divide by primes in (p, p+200/p].
    # This is fast!
    
    # Revised Strategy:
    # 1. Sieve primes up to 10^7.
    # 2. For each prime p, check j=1..200.
    #    For each j, check if p^2+j has a prime factor q > p.
    #    We only need to check primes q in (p, sqrt(p^2+j)].
    #    Since sqrt(p^2+j) is very close to p, the range is small.
    #    Specifically, q must be in (p, p + 200/p].
    #    For p=1000, range is (1000, 1000.2]. No primes.
    #    For p=10, range is (10, 12]. Primes: 11.
    #    So for large p, there are NO primes q > p such that q <= sqrt(p^2+j).
    #    Wait. If p^2+j has a factor q > p, then the other factor must be < p.
    #    So we just need to check if p^2+j has any factor q > p.
    #    This is equivalent to: p^2+j is NOT p-smooth.
    #    We can check this by dividing out all prime factors <= p.
    #    If the remainder is 1, it is p-smooth.
    #    If remainder > 1, it has a factor > p.
    
    # This is exactly what `is_p_smooth` does.
    # But we can optimize `is_p_smooth` by noting that for large p, the number of small primes is large.
    
    # Let's stick to the explicit verification but optimize the search space.
    # The inspiration used segmented sieve.
    # Let's use segmented sieve to generate primes up to 10^7.
    
    def segmented_sieve(low, high):
        if high < 2:
            return []
        low = max(low, 2)
        if low % 2 == 0:
            low += 1
        if high % 2 == 0:
            high -= 1
        
        limit = int(math.isqrt(high)) + 1
        small_primes = sieve(limit)
        
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        if low == 1:
            is_prime_seg[0] = 0
        if low == 2:
            is_prime_seg[0] = 1
        else:
            is_prime_seg[0] = 0
        
        if low % 2 != 0:
            for i in range(1, size, 2):
                is_prime_seg[i] = 0
        
        for p in small_primes:
            if p == 2:
                continue
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start - low, size, p):
                is_prime_seg[j] = 0
        
        result = []
        for i in range(size):
            if is_prime_seg[i]:
                result.append(low + i)
        return result
    
    # Phase 1: Check primes up to 10^5
    limit1 = 10**5
    primes1 = sieve(limit1)
    
    for p in primes1:
        if time.time() > deadline:
            break
        k = 0
        p2 = p * p
        while k < 200:
            n = p2 + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check primes in [10^5, 10^7] using segmented sieve
    current_low = limit1 + 1
    chunk_size = 10**5
    
    while current_low < 10**7:
        if time.time() > deadline:
            break
        
        current_high = current_low + chunk_size - 1
        primes_chunk = segmented_sieve(current_low, current_high)
        
        for p in primes_chunk:
            if time.time() > deadline:
                break
            
            # Optimization: If best_k is already high, we can stop early
            # if we find a failure at j <= best_k
            k = 0
            p2 = p * p
            while k < 200:
                n = p2 + k + 1
                if not is_p_smooth(n, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
        
        current_low = current_high + 1
    
    # Phase 3: Random search for larger primes if time permits
    if time.time() < deadline - 20:
        for _ in range(200):
            if time.time() > deadline:
                break
            x = random.randrange(10**7, 10**9)
            if x % 2 == 0:
                x += 1
            while not is_prime(x):
                x += 2
                if x > 10**9:
                    break
            if x > 10**9:
                continue
            k = 0
            p2 = x * x
            while k < 200:
                n = p2 + k + 1
                if not is_p_smooth(n, x):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = x

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
