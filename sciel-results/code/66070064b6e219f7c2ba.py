import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Deterministic Miller-Rabin for n < 3.3e24
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

    # Pollard's Brent cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(2, n - 1)
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

    def sieve_primes(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit+1, i):
                    is_p[j] = False
        return [i for i in range(2, limit+1) if is_p[i]]

    # Precompute small primes for batch division
    SMALL_PRIME_LIMIT = 200000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        temp = n
        for q in small_primes:
            if q > p or q * q > temp:
                break
            if temp % q == 0:
                while temp % q == 0:
                    temp //= q
                if temp == 1:
                    return True
        if temp == 1:
            return True
        if is_prime_mr(temp):
            return temp <= p
        try:
            f = pollard_brent(temp)
        except Exception:
            return False
        return is_p_smooth(f, p) and is_p_smooth(temp // f, p)
    
    def check_run(p, k_max=200):
        """Check how many consecutive integers p^2+1,...,p^2+k_max are p-smooth."""
        if time.time() - start_time > DEADLINE:
            return 0
        base = p * p
        k = 0
        for i in range(1, k_max + 1):
            if time.time() - start_time > DEADLINE:
                break
            if is_p_smooth(base + i, p):
                k = i
            else:
                break
        return k
    
    def fast_check(p, limit=20):
        """Quick check: trial division only for first 'limit' positions."""
        if time.time() - start_time > DEADLINE:
            return 0
        base = p * p
        count = 0
        for i in range(1, limit + 1):
            n = base + i
            temp = n
            for q in small_primes:
                if q > p or q * q > temp:
                    break
                if temp % q == 0:
                    while temp % q == 0:
                        temp //= q
                    if temp == 1:
                        break
            if temp == 1:
                count = i
            else:
                break
        return count

    best_k = 0
    best_p = 2

    def find_primes_segmented(lo, hi, max_count=1000):
        """Find up to max_count primes in [lo, hi] using Miller-Rabin."""
        primes = []
        if lo <= 2:
            primes.append(2)
            lo = 3
        if lo % 2 == 0:
            lo += 1
        count = 0
        for candidate in range(lo, hi + 1, 2):
            if count >= max_count:
                break
            if time.time() - start_time > DEADLINE:
                break
            if is_prime_mr(candidate):
                primes.append(candidate)
                count += 1
        return primes

    # Structural change: Two-phase search with fast pre-filtering
    # Phase 1: Scan large-to-small bands, fast-check each prime, only full-check promising ones
    # Phase 2: Local search around best found

    band_size = 50000000
    for band_start in range(1000000000, 1000000, -band_size):
        if time.time() - start_time > DEADLINE - 10:
            break
        if best_k >= 198:
            break
        lo = max(2, band_start - band_size + 1)
        hi = band_start
        primes = find_primes_segmented(lo, hi, 200)
        for p in primes:
            if time.time() - start_time > DEADLINE - 5:
                break
            fc = fast_check(p, 20)
            if fc <= best_k:
                continue
            k = check_run(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    break

    # Local search around best
    if best_k < 200 and time.time() - start_time < DEADLINE - 10:
        if best_p > 1000:
            lo = max(2, best_p - 100000)
            hi = best_p + 100000
            primes = find_primes_segmented(lo, hi, 300)
            for p in primes:
                if time.time() - start_time > DEADLINE - 5:
                    break
                if p == best_p:
                    continue
                fc = fast_check(p, 20)
                if fc <= best_k:
                    continue
                k = check_run(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break

    return {'p': best_p, 'k': best_k}
