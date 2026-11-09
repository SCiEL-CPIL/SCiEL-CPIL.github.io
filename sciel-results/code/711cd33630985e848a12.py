"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    
    # Precompute small primes up to 10^6 for fast trial division
    SMALL_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SMALL_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SMALL_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SMALL_LIMIT + 1) if sieve[i]]
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses for n < 2^64
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
    
    # Pollard's rho (Brent's variant)
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
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
                    if g > 1:
                        break
            if g != n:
                return g
    
    # Full factorization: returns list of prime factors
    def factorize(n):
        if n <= 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)
    
    # Check if n is p-smooth using small primes first, then full factorization
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Try small primes up to min(p, SMALL_LIMIT)
        for sp in small_primes:
            if sp > p:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        # Remaining factor: must be <= p and prime (or product of primes <= p)
        if n > p:
            factors = factorize(n)
            return all(f <= p for f in factors)
        else:
            return True
    
    # Quick check: how many consecutive values starting from p^2+1 are p-smooth?
    def quick_smooth_run(p, max_k=200):
        k = 0
        p_sq = p * p
        for i in range(1, max_k + 1):
            n = p_sq + i
            if n <= 1:
                continue
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > n:
                    break
                while n % sp == 0:
                    n //= sp
            if n == 1:
                k += 1
            elif n <= p:
                k += 1
            else:
                if is_prime_mr(n):
                    break
                else:
                    break
        return k
    
    # Full verification: verify that p^2+1 through p^2+k are all p-smooth
    def verify_run(p, k):
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not is_p_smooth(n, p):
                return False
        return True
    
    # Generate primes up to a limit using the sieve
    def primes_up_to(limit):
        if limit < 2:
            return []
        s = bytearray(b'\x01') * (limit + 1)
        s[0] = s[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if s[i]:
                s[i*i::i] = bytearray(len(s[i*i::i]))
        return [i for i in range(2, limit + 1) if s[i]]
    
    best_k = 0
    best_p = 2
    
    # ============================================================
    # CONGRUENCE-BASED PRE-SIEVE STRATEGY
    # ============================================================
    # For a candidate prime p, we need p^2 + j to be p-smooth for j = 1, ..., k.
    # If there exists a prime q > p such that q | (p^2 + j) for some j in [1, k],
    # then p^2 + j has a prime factor > p, so it's NOT p-smooth.
    #
    # Key insight: For fixed q, the condition q | (p^2 + j) means
    # p^2 ≡ -j (mod q), i.e., p^2 ≡ q - j (mod q).
    # So for each j, we can compute which residues p mod q make p^2 + j ≡ 0 (mod q).
    # If p falls in any of these "bad" residues for any j in [1, k_max],
    # then p cannot achieve a run of length k_max.
    #
    # We use this to pre-sieve candidate primes before doing expensive factorization.
    
    def build_congruence_sieve(prime_list, k_max=200):
        """
        For each prime q in prime_list (where q > some threshold),
        compute the set of residues p mod q such that p^2 + j ≡ 0 (mod q)
        for some j in [1, k_max].
        Returns a dict: q -> set of bad residues.
        """
        bad_residues = {}
        for q in prime_list:
            # For each j in [1, k_max], find p mod q such that p^2 ≡ -j (mod q)
            # i.e., p^2 ≡ q - j (mod q)
            residues = set()
            for j in range(1, k_max + 1):
                target = (q - j) % q
                # Find all x such that x^2 ≡ target (mod q)
                # Try x from 0 to q-1 (only feasible for small q)
                if q <= 10000:
                    for x in range(q):
                        if (x * x) % q == target:
                            residues.add(x)
                else:
                    # For larger q, use Tonelli-Shanks or just skip
                    # We'll only use small q for the pre-sieve
                    pass
            if residues:
                bad_residues[q] = residues
        return bad_residues
    
    def is_bad_prime(p, bad_residues, k_max=200):
        """
        Check if prime p is 'bad' for a run of length k_max,
        i.e., if for some j in [1, k_max], p^2 + j has a prime factor > p.
        We check this by seeing if p mod q is in the bad set for any q.
        """
        for q, residues in bad_residues.items():
            if p % q in residues:
                return True
        return False
    
    # Build the congruence sieve using small primes q
    # Use primes q from 100 to 5000 (small enough to enumerate residues)
    q_primes = [q for q in small_primes if 100 <= q <= 5000]
    print(f"Building congruence sieve with {len(q_primes)} primes q in [100, 5000]")
    bad_residues = build_congruence_sieve(q_primes, 200)
    print(f"Congruence sieve built: {len(bad_residues)} primes have bad residues")
    
    # Phase 1: Small primes up to 10^5 - use congruence pre-sieve
    small_search_limit = 100000
    primes_small = primes_up_to(small_search_limit)
    
    # First, find the best k without pre-sieve for small primes (baseline)
    for p in primes_small:
        if time.time() > deadline:
            break
        k = quick_smooth_run(p, 200)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p
    
    # Phase 2: Use congruence pre-sieve to filter candidates in larger ranges
    # For each candidate range, first check if the prime is "bad" using the congruence sieve
    # Only do quick_smooth_run on primes that pass the pre-sieve
    
    # Define is_prime_test for larger primes
    def is_prime_test(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
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
    
    # Search in chunks, using pre-sieve to skip bad candidates
    # Focus on ranges where the pre-sieve eliminates many candidates
    for start in range(100000, 5000000, 50000):
        if time.time() > deadline:
            break
        end = min(start + 50000, 5000000)
        tested = 0
        passed_presieve = 0
        for candidate in range(start, end, 2):
            if time.time() > deadline:
                break
            if not is_prime_test(candidate):
                continue
            tested += 1
            # Check congruence pre-sieve
            if is_bad_prime(candidate, bad_residues, 200):
                continue
            passed_presieve += 1
            # Only do expensive quick_smooth_run on candidates that pass pre-sieve
            k = quick_smooth_run(candidate, 200)
            if k > best_k:
                if verify_run(candidate, k):
                    best_k = k
                    best_p = candidate
        if tested > 0:
            print(f"Range [{start}, {end}]: tested {tested} primes, {passed_presieve} passed pre-sieve, best_k={best_k}")
    
    # Phase 3: Random sampling of larger primes with pre-sieve
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline and attempts < 3000:
            attempts += 1
            if random.random() < 0.5:
                candidate = random.randint(10**6, 10**7)
            elif random.random() < 0.5:
                candidate = random.randint(10**7, 10**8)
            else:
                candidate = random.randint(10**8, 10**9)
            
            if candidate % 2 == 0:
                candidate += 1
            
            for _ in range(100):
                if is_prime_test(candidate):
                    break
                candidate += 2
            else:
                continue
            
            # Pre-sieve check
            if is_bad_prime(candidate, bad_residues, 200):
                continue
            
            k = quick_smooth_run(candidate, 200)
            if k > best_k:
                if verify_run(candidate, k):
                    best_k = k
                    best_p = candidate
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
