"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 3.3e24 (more than enough for n < 1e18)
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses sufficient for n < 3.3e24
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
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
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
                    g = _gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def factorize(n):
        """Return list of prime factors of n."""
        if n <= 1:
            return []
        if is_prime(n):
            return [n]
        if n % 2 == 0:
            return [2] + factorize(n // 2)
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)

    def max_prime_factor(n):
        """Return the largest prime factor of n, or 0 if n==1."""
        if n <= 1:
            return 0
        if is_prime(n):
            return n
        factors = factorize(n)
        return max(factors)

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p. Early exit."""
        if n <= 1:
            return True
        # Quick check: if n <= p, all factors are <= p
        if n <= p:
            return True
        if is_prime(n):
            return n <= p
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                        41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for sp in small_primes:
            if sp > p:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        # Use Pollard-Brent to find a factor
        d = pollard_brent(n)
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def run_length(p, max_k=200):
        """Compute the run length: number of consecutive integers p^2+1, ..., p^2+k
        that are all p-smooth."""
        k = 0
        while k < max_k:
            n = p * p + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        return k

    # Sieve for small primes up to some limit
    def sieve(n):
        """Return list of primes up to n."""
        if n < 2:
            return []
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return [i for i in range(2, n + 1) if is_p[i]]

    # Phase 1: Search small primes (p <= 100000) with segmented sieve
    # For each prime p, check run length using is_p_smooth
    best_k = 0
    best_p = 2

    # Generate primes up to 200000 for the search
    primes = sieve(200000)

    for p in primes:
        if p > 100000:
            break
        if time.time() - start_time > DEADLINE:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes (100000 < p <= 10^9)
    # For large p, p^2 + i for i in 1..200 is around 1e10 to 1e10+200
    # The chance that all 200 numbers are p-smooth is very small for large p,
    # but we should check a range.
    # Strategy: for large p, most numbers p^2+1...p^2+200 will have a prime factor > p
    # with high probability. So run lengths will be short.
    # We sample primes in larger ranges.
    #
    # Actually, let's think about this more carefully.
    # For p ~ 10^6, p^2 ~ 10^12. The numbers p^2+1..p^2+200 are ~10^12.
    # A random number ~10^12 has about log(10^12)/log(2) ~ 40 bits, and its prime factors
    # are distributed. The largest prime factor of a random n is typically ~n^(1/2) or larger.
    # So for p ~ 10^6, we need all prime factors of p^2+i to be <= 10^6.
    # p^2+i ~ 10^12, so we need it to be 10^6-smooth, which means all prime factors <= 10^6.
    # The probability that a random 10^12 number is 10^6-smooth is roughly
    # rho(12/6) = rho(2) ~ 0.5 (Dickman function). Actually rho(2) = 1 - ln(2) ≈ 0.307.
    # So about 30% chance for one number. For 200 consecutive numbers, it's (0.307)^200 ≈ 0.
    # So for p ~ 10^6, run length 200 is essentially impossible.
    #
    # For p ~ 10^4, p^2 ~ 10^8. Need 10^4-smooth numbers around 10^8.
    # rho(8/4) = rho(2) ≈ 0.307. Same issue.
    #
    # For p ~ 10^3, p^2 ~ 10^6. Need 10^3-smooth around 10^6.
    # rho(6/3) = rho(2) ≈ 0.307.
    #
    # Hmm, so the probability doesn't change much with p. The run length is limited
    # by the probability of consecutive p-smooth numbers.
    #
    # Let's just search up to p=100000 in phase 1, and then do a few spot checks
    # for larger p to see if any give longer runs.
    #
    # Actually, for the task, the seed found k=2 with p in [3,2000]. Let's see if
    # we can do better with the improved factorization.

    # Phase 3: Check some larger primes
    # For p > 100000, generate random primes and check
    # But first, let's extend the sieve-based search to p up to 10^6
    # This might be too slow. Let's be selective.

    # For larger p, the run length is likely very short (0 or 1).
    # Let's check p in ranges [10^5, 2*10^5], [10^6, 2*10^6], etc.
    # but only a subset.

    # Actually, let's just check all primes up to 10^5 (already done) and then
    # do a random sample of larger primes.

    # Generate a few random primes in larger ranges
    # For p in [10^5, 10^9], check run length
    # But this could be very slow. Let's limit it.

    # Check primes in [100000, 500000]
    for p in primes:
        if p < 100000:
            continue
        if p > 500000:
            break
        if time.time() - start_time > DEADLINE:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Check some random primes in larger ranges
    random.seed(42)
    for _ in range(200):
        if time.time() - start_time > DEADLINE:
            break
        # Random prime between 10^6 and 10^9
        p = random.randint(10**6, 10**9)
        # Make it odd
        p |= 1
        # Find next prime >= p
        while not is_prime(p):
            p += 2
        if p > 10**9:
            continue
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': max(best_k, 1)}

# EVOLVE-BLOCK-END
