import math
import time

def construct():
    deadline = time.time() + 115.0

    def sieve(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes:
            if n == sp:
                return True
            if n % sp == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = 2, 1, 128
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
        return pollard_brent(n)

    SIEVE_LIMIT = 1000000
    primes = sieve(SIEVE_LIMIT)

    def all_factors_leq(n, p, primes_list, max_prime_idx):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Trial divide by primes <= min(p, SIEVE_LIMIT)
        for i in range(max_prime_idx + 1):
            q = primes_list[i]
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        if n <= 1:
            return True
        if n <= p:
            # n is prime (or 1), and <= p
            return True
        # n > p, need to check if n is prime and <= p (impossible since n > p)
        # or n is composite with all factors <= p
        # Since we divided by all primes <= min(p, SIEVE_LIMIT), 
        # any remaining factor > min(p, SIEVE_LIMIT)
        # If p <= SIEVE_LIMIT: n must be prime (proved earlier), and n > p, so False
        # If p > SIEVE_LIMIT: n could be composite with factors in (SIEVE_LIMIT, p]
        if p <= SIEVE_LIMIT:
            return False
        # p > SIEVE_LIMIT: try to factor n
        # n < p^2 + 200, and all prime factors > SIEVE_LIMIT
        # Try Pollard's rho
        factor = pollard_brent(n)
        # Check factor and n//factor
        if factor > p:
            return False
        if (n // factor) > p:
            return False
        return True

    best_k = 0
    best_p = 2

    # Check p = 2
    k = 0
    while k < 200:
        v = 4 + k + 1
        if v & (v - 1) == 0:  # power of 2
            k += 1
        else:
            break
    if k > best_k:
        best_k = k
        best_p = 2

    # Precompute index of last prime <= SIEVE_LIMIT (which is len(primes)-1)
    max_idx = len(primes) - 1

    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        
        p_sq = p * p
        k = 0
        p_idx = idx
        
        for k_val in range(1, 201):
            if time.time() > deadline:
                break
            v = p_sq + k_val
            if not all_factors_leq(v, p, primes, p_idx):
                break
            k = k_val
        
        if k > best_k:
            best_k = k
            best_p = p

    # Segmented sieve for primes beyond SIEVE_LIMIT
    def segmented_sieve(low, high, small_primes):
        size = high - low + 1
        is_prime = bytearray(b'\x01') * size
        for p in small_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime[j - low] = 0
        result = []
        for i in range(size):
            if is_prime[i]:
                val = low + i
                if val >= 2:
                    result.append(val)
        return result

    segment_size = 500000
    search_limit = min(10**9, 20000000)
    
    low = SIEVE_LIMIT + 1
    if low % 2 == 0:
        low += 1
    
    while low < search_limit:
        if time.time() > deadline:
            break
        high = min(low + segment_size - 1, search_limit)
        seg_primes = segmented_sieve(low, high, primes)
        
        for p in seg_primes:
            if time.time() > deadline:
                break
            if p > 10**9:
                break
            
            p_sq = p * p
            k = 0
            p_idx = max_idx  # all our primes are <= p
            
            for k_val in range(1, 201):
                if time.time() > deadline:
                    break
                v = p_sq + k_val
                if not all_factors_leq(v, p, primes, p_idx):
                    break
                k = k_val
            
            if k > best_k:
                best_k = k
                best_p = p
        
        low = high + 2

    return {'p': best_p, 'k': best_k}
