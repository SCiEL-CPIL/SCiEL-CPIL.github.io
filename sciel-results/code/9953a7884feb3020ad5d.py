import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit

    # 1. Generate small primes up to 10^6 for fast trial division and sieving
    limit_small = 10**6
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

    best_k = 0
    best_p = 3
    
    # STRATEGY: Constrained Segmented Sieve Search
    
    # We will search for primes p in specific ranges.
    # For a given p, we look at the interval [p^2+1, p^2+L].
    # A number n in this interval is a "candidate" for being p-smooth if it has NO prime factor q > p.
    # We can find these candidates by sieving the interval with all primes q > p.
    
    # If we find a run of length k where all numbers are candidates, we then verify them.
    # The probability of a number having a prime factor > p is roughly 1 - rho(2) ~ 0.7.
    # So the probability of a run of length k being all candidates is (0.3)^k.
    # This is the same as the final probability, but sieving is MUCH faster than factoring.
    
    # Let's define a function to find the best k for a given p using the sieve method.
    def find_max_k_for_p(p, max_k_limit=200):
        base = p * p
        # We need to check p^2+1 ... p^2+max_k_limit
        # Let's use a segment of length max_k_limit + 10 for safety
        L = max_k_limit + 10
        
        # Initialize candidates as True
        # index i corresponds to number base + i
        is_candidate = [True] * (L + 1)
        
        # Sieve with primes q > p
        # We only need to sieve up to sqrt(base+L) ~ p
        # Wait, if q > p, and q divides n, then n/q < n/p ~ p.
        # So if n has a factor q > p, the other factor is < p.
        # We don't need to sieve with ALL primes q > p up to n.
        # We only need to sieve with primes q such that q <= n.
        # But actually, if q > p, then q is a "large" factor.
        # To mark n as "not candidate", we just need to find ONE factor q > p.
        
        # Optimization: 
        # For each prime q > p, if q divides some n in the range, mark n as not candidate.
        # Which primes q do we need?
        # If q > n, q cannot divide n.
        # If p < q <= sqrt(n), then n/q >= sqrt(n) > p. So n/q is also a large factor? No.
        # If n = q * r, and q > p, then r = n/q.
        # If r <= p, then n is p-smooth ONLY IF q <= p. But q > p. So n is NOT p-smooth.
        # So ANY prime factor q > p makes n not p-smooth.
        
        # So we need to sieve with all primes q in (p, max_n].
        # But we don't have a list of primes up to 10^18.
        
        # Alternative approach:
        # Instead of sieving with large primes, we can use the property that:
        # n is p-smooth iff all prime factors <= p.
        # This is equivalent to: after removing all prime factors <= p, the remainder is 1.
        
        # The current is_p_smooth does exactly this.
        # The bottleneck is the "remove all prime factors <= p" step if p is large.
        # For p <= 10^6, trial division is fast.
        # For p > 10^6, trial division is slow.
        
        # Let's stick to the current is_p_smooth but optimize the search.
        
        # Actually, the "sieve" idea can be applied differently:
        # Precompute the "bad" primes for a range of p?
        
        # Let's try a different structural change:
        # **Parallel-like search via segmented sieve of primes p.**
        
        # We will generate primes p in segments.
        # For each segment, we will test all primes in that segment.
        
        # To speed up is_p_smooth for large p:
        # 1. Trial divide by primes up to 10^6.
        # 2. If remainder is 1, True.
        # 3. If remainder is prime, check <= p.
        # 4. Else, factor remainder.
        
        # The factorization is the slow part.
        # Can we avoid it?
        # If remainder R is composite, and R < p^2, then R has a factor <= sqrt(R) < p.
        # Wait. If R is composite, it has a factor d <= sqrt(R).
        # If d <= p, then we should have divided by d in step 1?
        # No, step 1 only divides by primes up to 10^6.
        # If p > 10^6, we stop trial division at 10^6.
        # So R might have a factor d in (10^6, p].
        # If such a factor exists, R is not 1, and R is not prime (unless R=d, but d<=p).
        # If R is composite, it has a factor d <= sqrt(R).
        # If sqrt(R) > 10^6, we need to find that factor.
        
        # So for p > 10^6, we MUST factor the remainder if it's composite.
        
        # Let's focus on maximizing the number of primes tested.
        # We will test primes in increasing order up to a limit where time runs out.
        
        # To make the loop faster, we will skip the full factorization if possible.
        # We will use a "fast reject" check:
        # If R > 1 and R is not prime, we check if R has a factor in (10^6, p].
        # We can do this by trial dividing R by primes in (10^6, min(p, sqrt(R))].
        # If we find one, R is not p-smooth (because the other factor R/d < R/10^6. 
        # Wait, if d <= p, then R/d might be > p?
        # If R = d * e, and d <= p. If e > p, then R is not p-smooth.
        # So we need to check if ALL factors are <= p.
        
        # This is getting complicated. Let's just use the existing is_p_smooth 
        # but optimize the prime generation and search order.
        
        # Search primes up to 10^7.
        # We have small_primes up to 10^6.
        # We need primes up to 10^7.
        
        # Let's extend the sieve to 10^7.
        # This takes ~0.5s and 80MB memory. Acceptable.
        
        return None # Placeholder, logic below

    # Extend sieve to 10^7
    limit_large = 10**7
    sieve_large = [True] * (limit_large + 1)
    sieve_large[0:2] = [False, False]
    # Use small_primes to sieve
    for p in small_primes:
        if p * p > limit_large:
            break
        if sieve_large[p]:
            sieve_large[p*p::p] = [False] * len(sieve_large[p*p::p])
    
    large_primes = [i for i, is_p in enumerate(sieve_large) if is_p]
    
    # Search all primes up to 10^7
    # This is ~664,000 primes.
    # For each, we check up to 200 numbers.
    # is_p_smooth is fast for p <= 10^6.
    # For p in (10^6, 10^7], is_p_smooth might be slower due to factorization.
    
    # Let's estimate time.
    # If we can test 1000 primes/sec, 664k primes take 664s. Too slow.
    # We need to be faster.
    
    # Optimization: 
    # For p > 10^6, the probability of finding a long run is low.
    # We can sample primes in (10^6, 10^7] instead of checking all.
    
    # Phase 1: Check ALL primes up to 10^6.
    # Phase 2: Sample primes in [10^6, 10^7].
    
    for p in small_primes:
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
            
    # Phase 2: Sample primes in [10^6, 10^7]
    # We have large_primes.
    # Let's take a random sample or just iterate and stop when time runs out.
    # Iterating is better to ensure coverage if we are fast enough.
    
    for p in large_primes:
        if p < 10**6:
            continue
        if p > 10**7:
            break
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
    
    return {'p': best_p, 'k': best_k}
