"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 3.3e24
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

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
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

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        if n <= p:
            return True
        if is_prime(n):
            return n <= p
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                        41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                        101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                        151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
        for sp in small_primes:
            if sp > p:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        d = pollard_brent(n)
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def sieve(n):
        """Return list of primes up to n."""
        if n < 2:
            return []
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, n + 1) if is_p[i]]

    def run_length_via_sieve(p, max_k=200):
        """
        Use segmented sieve over [p^2+1, p^2+max_k] to find the run length.
        For each number in the window, we remove all prime factors <= p.
        If any number has residual > 1, it has a prime factor > p.
        Returns the length of the longest prefix where all numbers are p-smooth.
        """
        p2 = p * p
        n = max_k  # window size
        
        # Create an array of the numbers p^2+1, ..., p^2+max_k
        # We'll track the residual after removing all prime factors <= p
        residuals = [p2 + i for i in range(1, n + 1)]
        
        # Sieve with all primes <= p
        # We need primes up to min(p, sqrt(p^2 + max_k))
        # Since p^2 + max_k < (p+1)^2 for p > max_k, we need primes up to p
        # But we can limit: any prime factor of p^2+i that is > sqrt(p^2+max_k) 
        # must appear only once. For our purpose, we sieve with primes <= p.
        
        # Get primes up to p (but p can be up to 10^9, so we need to be smart)
        # For the sieve, we only need primes up to min(p, sqrt(p^2 + max_k))
        # sqrt(p^2 + max_k) ≈ p + max_k/(2p), so for large p, it's just above p
        # Actually, we need primes up to p to fully factorize p^2+i
        # But p can be huge. Let's think differently.
        
        # Key insight: For the number n = p^2 + i (1 <= i <= 200),
        # any prime factor q of n satisfies q <= n = p^2 + i < p^2 + p^2 = 2*p^2
        # So q < 2*p^2, meaning q can be up to ~2*p^2.
        # But we want to know if q > p. 
        # If we remove all prime factors <= p, the residual is either 1 (p-smooth)
        # or a product of primes > p. Since n < 2*p^2, the residual can have
        # at most one prime factor > p (because if it had two, their product > p^2,
        # and combined with factors <= p... actually let's just check residual > 1).
        
        # Strategy: sieve with primes up to p. But generating primes up to p=10^9 is infeasible.
        # Instead: for each number in the window, try to factor it.
        # But that's what we're trying to avoid.
        
        # Better approach: For each prime q <= p, mark which numbers in the window
        # are divisible by q, and divide them out.
        # We only need primes q where q divides some p^2 + i for 1 <= i <= 200.
        # p^2 + i ≡ 0 (mod q) means p^2 ≡ -i (mod q).
        # So q must be such that -i is a quadratic residue mod q for some i in 1..200.
        
        # Simpler: just iterate over small primes and divide.
        # For large p, most numbers will have a large prime factor, so the run will be short.
        # We can early-exit.
        
        # Let's use a hybrid: try small primes via sieve, then for residuals, use factorization.
        
        # Generate primes up to some limit for the sieve part
        sieve_limit = min(p, 100000)
        if sieve_limit >= 2:
            small_primes_list = sieve(sieve_limit)
            for q in small_primes_list:
                if q > p:
                    break
                # Find the first index in residuals where (p^2 + idx) % q == 0
                # i.e., find i in 1..n such that (p^2 + i) % q == 0
                # p^2 % q = r, so we need i ≡ -r (mod q)
                r = (p2 % q)
                target = (-r) % q
                if target == 0:
                    start_i = q  # i = q, 2q, ...
                else:
                    start_i = target
                # i ranges from 1 to n
                if start_i > n:
                    continue
                for idx in range(start_i - 1, n, q):
                    residuals[idx] //= q
                    # Keep dividing
                    while residuals[idx] % q == 0:
                        residuals[idx] //= q
        
        # Now check residuals
        # For each residual, if it's > 1, check if it's <= p (then it's a prime factor <= p)
        # or > p (then it has a factor > p)
        # Actually, the residual after dividing by all primes <= min(p, sieve_limit)
        # might still have prime factors between sieve_limit and p.
        # So we need to check if the residual is p-smooth.
        
        k = 0
        for i in range(n):
            res = residuals[i]
            if res <= 1:
                k += 1
                continue
            # Check if res is p-smooth
            # res could be prime, or composite with factors > sieve_limit
            if res <= p:
                # All factors are <= res <= p, so p-smooth
                k += 1
            else:
                # res > p, need to check if it has a factor > p
                # Since we've removed all factors <= sieve_limit,
                # res has all prime factors > sieve_limit
                # If res is prime and > p, not p-smooth
                # If res is composite, it could be product of primes in (sieve_limit, p]
                # or have a prime > p
                # Check: is res p-smooth?
                if is_p_smooth(res, p):
                    k += 1
                else:
                    break
        return k

    def run_length(p, max_k=200):
        """Compute run length using the sieve-based approach."""
        if p <= 100000:
            # For small p, use the sieve approach
            return run_length_via_sieve(p, max_k)
        else:
            # For large p, the numbers p^2+1..p^2+200 are ~p^2
            # The probability that a random number ~p^2 is p-smooth is rho(2) ≈ 0.307
            # So run length 1 has prob ~0.3, run length 2 has prob ~0.09, etc.
            # Use direct factorization for each number until we find a non-p-smooth one
            k = 0
            p2 = p * p
            while k < max_k:
                n = p2 + k + 1
                if not is_p_smooth(n, p):
                    break
                k += 1
            return k

    # Main search
    best_k = 0
    best_p = 2

    # Phase 1: Search all primes up to 200000
    primes = sieve(200000)
    
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search primes in [200000, 1000000]
    # Generate primes in this range using segmented sieve or just extend
    if time.time() - start_time < DEADLINE:
        # Generate primes up to 1000000
        primes_1m = sieve(1000000)
        for p in primes_1m:
            if p < 200000:
                continue
            if time.time() - start_time > DEADLINE:
                break
            k = run_length(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 3: Random search for larger primes
    if time.time() - start_time < DEADLINE - 10:
        random.seed(12345)
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() - start_time < DEADLINE - 5:
            # Pick a random prime in a larger range
            # Focus on ranges where p^2 is not too large
            # For p in [10^6, 10^7], p^2 in [10^12, 10^14]
            # For p in [10^7, 10^8], p^2 in [10^14, 10^16]
            # The Dickman rho(2) ≈ 0.307 doesn't depend on magnitude
            # So run lengths are similar in expectation
            lo = random.choice([10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7, 10**8])
            hi = min(lo * 10, 10**9)
            p = random.randint(lo, hi)
            p |= 1  # make odd
            # Find next prime
            for _ in range(1000):
                if p > 10**9:
                    break
                if is_prime(p):
                    break
                p += 2
            if p > 10**9:
                attempts += 1
                continue
            k = run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
            attempts += 1

    return {'p': best_p, 'k': max(best_k, 1)}

# EVOLVE-BLOCK-END
