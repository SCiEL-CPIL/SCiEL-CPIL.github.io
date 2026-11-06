import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
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
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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
                        q = (q * abs(x - y)) % n
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

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return p
                if is_prime(n):
                    return n
        factors = []
        while n > 1:
            if is_prime(n):
                factors.append(n)
                break
            d = pollard_brent(n)
            if d == n:
                continue
            factors.append(d)
            n //= d
        return max(factors)

    # Pre-compute small primes for trial division speedup
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]

    # Fast check: does n have any prime factor > p?
    # Key optimization: if after removing small factors, remainder > p,
    # then either the remainder itself is prime > p (bad) or composite with factors > p (bad).
    # We only need Pollard-Brent if remainder <= p (to verify it's fully p-smooth).
    def has_large_factor(n, p):
        if n <= 1:
            return False
        if p < 2:
            return True
        # Trial divide by small primes up to min(p, limit)
        for sp in small_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
        if n == 1:
            return False
        # If remainder > p, it must contain a prime factor > p
        # (since all its prime factors are > max small prime used, and if all were <= p,
        #  we would have divided them out... actually no, we only divided by primes <= limit)
        # Wait: if p > limit, we divided by all primes <= limit. The remainder has prime factors > limit.
        # If p >= limit and remainder > p, then remainder has a factor > p? 
        # Not necessarily: remainder could be composite with factors in (limit, p].
        # But if remainder > p and all its prime factors are <= p, then it's p-smooth.
        # We can't conclude immediately. However, if remainder is PRIME and > p, that's a large factor.
        # For composite remainder > p, we need to check.
        # 
        # Actually, a simpler sufficient condition: if remainder > p, we CANNOT conclude it's bad
        # unless we verify. But if remainder is prime > p, it IS bad.
        # Let's check if n is prime first (fast for many cases).
        if is_prime(n):
            return n > p
        # n is composite. Find largest prime factor.
        return largest_prime_factor(n) > p

    def has_large_factor_fast(n, p):
        """Faster version: early exit if n > p and n is prime."""
        if n <= 1:
            return False
        if p < 2:
            return True
        # Quick check: if n itself is prime
        if is_prime(n):
            return n > p
        # Trial divide by small primes
        for sp in small_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
        if n == 1:
            return False
        # After removing small factors, if n > p, check if prime
        if is_prime(n):
            return n > p
        # Composite: need full factorization
        return largest_prime_factor(n) > p

    def check_run_length(p, max_k=200, use_fast=True):
        """Check how many consecutive values p^2+1, ..., p^2+k are p-smooth."""
        p2 = p * p
        k = 0
        checker = has_large_factor_fast if use_fast else has_large_factor
        while k < max_k:
            val = p2 + k + 1
            if checker(val, p):
                break
            k += 1
        return k

    # Segmented sieve to generate primes in a range
    def generate_primes_in_range(low, high, base_primes):
        """Generate primes in [low, high] using segmented sieve."""
        if high < low:
            return []
        size = high - low + 1
        is_prime_seg = [True] * size
        is_prime_seg[0] = (low >= 2)
        for p in base_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = False
        if low == 1:
            is_prime_seg[0] = False
        return [low + i for i in range(size) if is_prime_seg[i]]

    # Generate base primes up to sqrt(10^9) ~ 31623
    base_limit = 31623
    base_sieve = [True] * (base_limit + 1)
    base_sieve[0] = base_sieve[1] = False
    for i in range(2, int(base_limit**0.5) + 1):
        if base_sieve[i]:
            for j in range(i*i, base_limit + 1, i):
                base_sieve[j] = False
    base_primes = [i for i, is_p in enumerate(base_sieve) if is_p]

    best_k = 0
    best_p = 2

    # Phase 1: Screen many candidates with a quick check (k up to 5)
    # Phase 2: Deep check (k up to 200) on promising candidates
    
    # Strategy: use segmented sieve to scan ranges, check each prime
    # Start with smaller ranges for quick wins, then expand
    
    # First, try a structured scan of primes in various ranges
    # We'll use segments of size ~100000 for efficiency
    
    SEGMENT_SIZE = 100000
    
    # Define search ranges to cover [10^5, 10^9]
    # We'll prioritize: first do a broad screen, then deep-dive on hits
    
    # Phase 1: Broad screen - check k up to 3 for many primes
    # This is fast because we stop early
    
    # Generate segments across the range
    search_ranges = [
        (10**5, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
    ]
    
    # For each range, scan in segments
    # Keep track of candidates that pass k>=3 for deep checking
    
    candidate_primes = []  # primes that passed initial screen
    
    # Time budget: spend ~60% on screening, ~40% on deep verification
    screen_deadline = start_time + (DEADLINE - start_time) * 0.55
    
    for range_low, range_high in search_ranges:
        if time.time() > screen_deadline:
            break
        
        # Scan this range in segments
        seg_start = range_low
        while seg_start < range_high:
            if time.time() > screen_deadline:
                break
            
            seg_end = min(seg_start + SEGMENT_SIZE - 1, range_high)
            primes_in_seg = generate_primes_in_range(seg_start, seg_end, base_primes)
            
            for p in primes_in_seg:
                if time.time() > screen_deadline:
                    break
                # Quick screen: check k up to 5
                k_quick = check_run_length(p, max_k=5, use_fast=True)
                if k_quick >= 3:
                    candidate_primes.append((p, k_quick))
            
            seg_start = seg_end + 1
    
    # Phase 2: Deep check on candidates
    # Sort by initial k (descending) to prioritize promising ones
    candidate_primes.sort(key=lambda x: x[1], reverse=True)
    
    for p, k_initial in candidate_primes:
        if time.time() > DEADLINE:
            break
        # Deep check up to k=200
        k_full = check_run_length(p, max_k=200, use_fast=True)
        if k_full > best_k:
            best_k = k_full
            best_p = p
    
    # If no candidates found, do a random search fallback
    if best_k == 0:
        while time.time() < DEADLINE:
            # Generate a random prime
            found_prime = False
            for _ in range(50):
                if random.random() < 0.3:
                    p = random.randint(10**5, 10**6)
                elif random.random() < 0.5:
                    p = random.randint(10**6, 10**7)
                else:
                    p = random.randint(10**7, 10**9)
                p |= 1
                if is_prime(p):
                    found_prime = True
                    break
            if not found_prime:
                continue
            
            k = check_run_length(p, max_k=200, use_fast=True)
            if k > best_k:
                best_k = k
                best_p = p
    
    # Also check some very small primes just in case
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > DEADLINE:
            break
        k = check_run_length(p, max_k=200, use_fast=True)
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
