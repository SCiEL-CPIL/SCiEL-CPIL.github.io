import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin for 120s limit

    # 1. Generate small primes up to 100,000 for fast trial division and sieving
    limit_small = 100000
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Helper: Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        # Check small primes
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                return n == p
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
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

    # Helper: Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
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
        return g

    def factor(n):
        factors = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    # Helper: Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p):
        if n == 1:
            return True
        
        temp = n
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        
        if temp == 1:
            return True
        
        # If temp is prime, check if it's <= p
        if is_prime_mr(temp):
            return temp <= p
        
        # If temp is composite, factor it
        try:
            factors = factor(temp)
            for f in factors:
                if f > p:
                    return False
            return True
        except:
            return False

    # Helper: Fast approximate check for p-smoothness
    # This is much faster than is_p_smooth but less accurate.
    # It only checks divisibility by primes up to limit_check.
    def is_p_smooth_approx(n, p, limit_check):
        if n == 1:
            return True
        
        temp = n
        for q in small_primes:
            if q > p:
                break
            if q > limit_check:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        
        # If temp is 1, it's definitely p-smooth.
        if temp == 1:
            return True
        
        # If temp is prime, check if it's <= p.
        # Note: This might be a false negative if temp is composite with large factors,
        # but it's a good heuristic for pre-sieving.
        if is_prime_mr(temp):
            return temp <= p
        
        # If temp is composite, it's likely not p-smooth if it has factors > limit_check.
        # We can be a bit more aggressive here.
        # If temp > 1 and is not prime, it has a factor > limit_check (since we divided out all <= limit_check).
        # If that factor is > p, it's not p-smooth.
        # If that factor is <= p, it might be p-smooth.
        # To be safe, we return False if temp is not 1 and not prime.
        # This might discard some valid candidates, but it's fast.
        return False

    best_k = 0
    best_p = 3
    
    # STRATEGY: Probabilistic Pre-Sieving
    
    # We want to search primes p up to 10^9.
    # We can't check all primes in that range.
    # We will sample primes in various ranges.
    
    # Phase 1: Check all primes up to 10^6.
    # We have small_primes up to 100,000.
    # Let's extend the sieve to 10^6.
    
    limit_large = 10**6
    sieve_large = [True] * (limit_large + 1)
    sieve_large[0:2] = [False, False]
    for p in small_primes:
        if p * p > limit_large:
            break
        if sieve_large[p]:
            sieve_large[p*p::p] = [False] * len(sieve_large[p*p::p])
    large_primes = [i for i, is_p in enumerate(sieve_large) if is_p]
    
    # Check all primes up to 10^6
    for p in large_primes:
        if time.time() > deadline:
            break
            
        base = p * p
        k = 0
        for offset in range(1, 201):
            if time.time() > deadline:
                break
            n = base + offset
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Sample primes in [10^6, 10^9]
    # We will sample primes in logarithmic intervals.
    # For each interval [L, R], we will sample some primes.
    
    # To find primes in [L, R], we can use a segmented sieve or random search.
    # Random search is simpler and good enough for sampling.
    
    def find_prime_near(x, max_attempts=100):
        # Find a prime near x
        for _ in range(max_attempts):
            # Random offset
            offset = random.randint(-1000, 1000)
            y = x + offset
            if y < 2:
                continue
            if is_prime_mr(y):
                return y
        return None
    
    # Intervals to sample
    # We will sample more densely in smaller ranges and less densely in larger ranges.
    # The probability of a long run is roughly constant, so we should sample proportionally to the number of primes.
    # Number of primes in [L, R] is approx (R-L)/ln(R).
    
    # Let's define a list of (L, R, num_samples)
    # We will adjust num_samples based on available time.
    
    intervals = [
        (10**6, 10**7, 500),
        (10**7, 10**8, 1000),
        (10**8, 10**9, 2000),
    ]
    
    # We will use a fast pre-sieve to filter candidates before doing the full check.
    # Pre-sieve: Check if p^2+1 ... p^2+10 are p-smooth using is_p_smooth_approx.
    # If any of them fails the approx check, skip the full check.
    
    for L, R, num_samples in intervals:
        if time.time() > deadline:
            break
            
        # Estimate time per sample
        # Full check: ~200 * is_p_smooth. is_p_smooth is fast for p <= 10^6, slower for larger.
        # Pre-sieve: ~10 * is_p_smooth_approx.
        
        # We will try to fit as many samples as possible in the remaining time.
        # Let's assume each full check takes ~0.01s (10ms).
        # Each pre-sieve takes ~0.001s (1ms).
        # So one sample (pre-sieve + maybe full check) takes ~0.011s.
        # In 100s, we can do ~9000 samples.
        
        # We will sample num_samples primes, but stop if time runs out.
        
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            
            # Pick a random number in [L, R]
            x = random.randint(L, R)
            p = find_prime_near(x)
            if p is None or p < L or p > R:
                continue
            
            # Pre-sieve: Check if p^2+1 ... p^2+10 are p-smooth using is_p_smooth_approx.
            # If any fails, skip the full check.
            base = p * p
            pre_sieve_ok = True
            for offset in range(1, 11):
                n = base + offset
                if not is_p_smooth_approx(n, p, 10000):
                    pre_sieve_ok = False
                    break
            
            if not pre_sieve_ok:
                continue
            
            # Full check: Check if p^2+1 ... p^2+200 are p-smooth using is_p_smooth.
            k = 0
            for offset in range(1, 201):
                if time.time() > deadline:
                    break
                n = base + offset
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
