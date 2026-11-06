import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit

    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            r += 1
            d //= 2
        for a in MR_BASES:
            if a >= n:
                continue
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
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

    # Precompute small primes for trial division
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Generate primes up to 10^6 for trial division and small prime search
    TRIAL_DIV_LIMIT = 1000000
    small_primes = sieve_primes(TRIAL_DIV_LIMIT)
    
    # Separate list for very small primes for fast initial division
    # We will use the first ~100 primes for quick division
    quick_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        
        # 1. Trial division by very small primes
        for q in quick_primes:
            if q > p:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
        
        if n == 1:
            return True
        
        # 2. If remaining n is small enough, check directly
        if n <= p:
            return True
            
        # 3. Check if n is prime
        if is_prime_mr(n):
            return False # Since n > p

        # 4. Factorize n using Pollard-Brent
        # We only need to check if factors are <= p
        d = pollard_brent(n)
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 2:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Exhaustive search for small primes
    # Primes up to 100,000 are cheap to check.
    # We have primes up to 1,000,000 precomputed.
    for p in small_primes:
        if p > 100000:
            break
        if time.time() > DEADLINE - 10:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 20: # Good improvement, keep looking
                pass

    # Phase 2: Search medium primes (100k to 1M)
    if time.time() < DEADLINE - 15:
        for p in small_primes:
            if p <= 100000:
                continue
            if p > 1000000:
                break
            if time.time() > DEADLINE - 5:
                break
            # Throttle: check every 10th prime in this range to save time
            # Or just check all if fast enough. 78,000 primes.
            # is_p_smooth is fast for small p.
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 3: Random search for large primes
    if time.time() < DEADLINE - 10:
        # Focus on ranges where p is large but not too large
        # p^2 < 10^18 is fine for Python ints.
        search_ranges = [
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9)
        ]
        
        trials_per_range = 500
        for lo, hi in search_ranges:
            if time.time() > DEADLINE - 5:
                break
            for _ in range(trials_per_range):
                if time.time() > DEADLINE - 5:
                    break
                x = random.randrange(lo, hi)
                if x % 2 == 0:
                    x += 1
                # Find next prime
                found = False
                for _ in range(50):
                    if is_prime_mr(x):
                        found = True
                        break
                    x += 2
                if not found or x > hi or x > 10**9:
                    continue
                
                k = check_run(x)
                if k > best_k:
                    best_k = k
                    best_p = x

    # Phase 4: Local search around best_p
    if time.time() < DEADLINE - 5 and best_p > 1000:
        center = best_p
        lo = max(2, center - 50000)
        hi = min(10**9, center + 50000)
        
        x = lo
        if x % 2 == 0:
            x += 1
        checked = 0
        while x <= hi and checked < 2000:
            if time.time() > DEADLINE - 2:
                break
            if is_prime_mr(x):
                k = check_run(x)
                if k > best_k:
                    best_k = k
                    best_p = x
            x += 2
            checked += 1

    if best_k < 1:
        best_k = 1
        best_p = 2

    return {'p': best_p, 'k': best_k}
