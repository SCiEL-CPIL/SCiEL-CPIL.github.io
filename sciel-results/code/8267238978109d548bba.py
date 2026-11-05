import math
import time
import random
import sys

# Deterministic Miller-Rabin bases for n < 2^64
MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
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
        m = random.randrange(1, n)
        g = 1
        r = 1
        q = 1
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

def factorize(n, limit=None):
    """
    Returns list of prime factors.
    If limit is provided, we only care if all factors are <= limit.
    Returns None if a factor > limit is found.
    """
    if n == 1:
        return []
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def _factorize_rec(m, limit):
        if m == 1:
            return []
        
        # Trial division by small primes
        for p in small_primes:
            if p * p > m:
                break
            if m % p == 0:
                if limit is not None and p > limit:
                    return [p] # Failed
                factors = [p]
                rest = m // p
                while rest % p == 0:
                    factors.append(p)
                    rest //= p
                rest_factors = _factorize_rec(rest, limit)
                if rest_factors is None:
                    return None
                return factors + rest_factors
        
        if m == 1:
            return []
        
        # Check if m is prime
        if is_prime(m):
            if limit is not None and m > limit:
                return [m] # Failed
            return [m]
        
        # Pollard Brent
        d = pollard_brent(m)
        if d == m:
            # Retry or fallback
            return _factorize_rec(m, limit)
        factors1 = _factorize_rec(d, limit)
        if factors1 is None:
            return None
        factors2 = _factorize_rec(m // d, limit)
        if factors2 is None:
            return None
        return factors1 + factors2

    return _factorize_rec(n, limit)

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n == 1:
        return True
    
    if is_prime(n):
        return n <= p
    
    d = pollard_brent(n)
    if d == n:
        return is_p_smooth(n, p)
    
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def sieve(n):
    """
    Sieve of Eratosthenes.
    Returns list of primes up to n.
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Strategy:
    # 1. Sieve primes up to a limit (e.g., 10^6) to use as "obstructions".
    # 2. Scan candidate primes p in a large range (e.g., 10^7 to 10^8).
    # 3. For each candidate p, use a "residue sieve" to find the max run k.
    #    The residue sieve is much faster than factoring p^2+i for i=1..k.
    #    It checks if p^2 + i has any prime factor > p.
    #    A number n is p-smooth iff it has no prime factor > p.
    #    For n = p^2 + i, any prime factor q > p must satisfy q | (p^2 + i).
    #    This implies p^2 ≡ -i (mod q).
    #    We can precompute for each prime q > p (from our sieve list) the residues i 
    #    such that p^2 + i ≡ 0 (mod q). If i is in [1, k_max], then p^2+i is NOT p-smooth.
    #    Wait, q > p. If q | (p^2 + i), then p^2 + i >= q.
    #    Since i <= 200, p^2 + i is close to p^2.
    #    If q > p, then q cannot divide p^2 + i unless p^2 + i >= q.
    #    Actually, if q | (p^2 + i), then p^2 + i = m*q.
    #    Since i is small, p^2 + i is roughly p^2.
    #    If q > p, then m must be small? No, q can be large.
    #    But if q > p, then q is a "large" prime factor.
    #    We want to find if there exists any prime q > p such that q | (p^2 + i) for i in 1..k.
    #    
    #    Let's refine the "Residue Sieve" idea:
    #    We have a list of primes Q = {q1, q2, ...} where q_j > p.
    #    For a fixed p, we want to find the smallest i in 1..200 such that p^2 + i has a prime factor > p.
    #    This is equivalent to finding the smallest i such that p^2 + i is NOT p-smooth.
    #    
    #    How to check if p^2 + i has a prime factor > p?
    #    Let N = p^2 + i.
    #    If N is prime, then N > p (since p^2+1 > p for p>=2). So it fails.
    #    If N is composite, it might have a factor > p.
    #    
    #    The "Residue Sieve" approach:
    #    We can't just sieve i directly because the condition "has a factor > p" is global.
    #    However, we can use the primes from our sieve (up to 10^6) as a filter.
    #    If p is large (e.g., 10^8), then any prime factor q of p^2+i that is > p must be > 10^8.
    #    But p^2+i is approx 10^16.
    #    If p^2+i has a factor q > p, then the cofactor c = (p^2+i)/q < p.
    #    So p^2+i = q * c where c < p and q > p.
    #    This means p^2+i is divisible by some c < p.
    #    If we know all prime factors of p^2+i that are <= p, we can determine the rest.
    #    
    #    Actually, a simpler and more robust strategy for large p:
    #    Just use the fast factorization check `is_p_smooth` but optimize the search space for p.
    #    The current program scans p from 2 upwards. This is slow because small p have small p^2,
    #    so p^2+i is small and likely has small factors, but also the run length k is limited by the density of primes.
    #    For large p, p^2 is large.
    #    
    #    Let's stick to the current factorization method but change the SEARCH STRATEGY for p.
    #    Current: p = 2, 3, 5, ... (ascending)
    #    New: Sample p from a large range (e.g., 10^7 to 10^8) using random sampling or a segmented sieve.
    #    Why? The probability of a long run might be higher for larger p? Or at least the search space is larger.
    #    Also, for large p, the "failure" condition (finding a prime factor > p) might be easier to detect?
    #    No, factorization is harder for larger numbers.
    #    
    #    Let's look at the "Residue Sieve" again.
    #    If we fix p, we want to find the max k such that for all i in 1..k, p^2+i is p-smooth.
    #    This is equivalent to: for all i in 1..k, p^2+i has no prime factor > p.
    #    
    #    Consider the primes q > p.
    #    If q | (p^2 + i), then i ≡ -p^2 (mod q).
    #    Since 1 <= i <= 200, we only care if -p^2 (mod q) is in [1, 200].
    #    If for some i in [1, 200], there exists a prime q > p such that i ≡ -p^2 (mod q), then p^2+i has a factor q > p.
    #    Wait, this is only true if q <= p^2+i. Which is always true for q > p if p^2+i >= q.
    #    Since p^2+i > p^2 > p, and q > p, it's possible q > p^2+i? No, q is a factor, so q <= p^2+i.
    #    So yes, if i ≡ -p^2 (mod q) for some prime q > p, then q divides p^2+i.
    #    And since q > p, p^2+i is NOT p-smooth.
    #    
    #    So, for a fixed p, the "bad" residues i are those that satisfy i ≡ -p^2 (mod q) for some prime q > p.
    #    We want to find the smallest i in 1..200 that is NOT bad.
    #    Wait, we want the largest k such that 1..k are ALL good.
    #    So we want to find the smallest i in 1..200 that IS bad. Let this be i_min.
    #    Then k = i_min - 1.
    #    
    #    How to find i_min?
    #    We need to check for each i in 1..200 if there exists a prime q > p such that i ≡ -p^2 (mod q).
    #    This is equivalent to checking if p^2 + i has a prime factor > p.
    #    This is exactly what `is_p_smooth` does, but we can be smarter.
    #    
    #    Instead of factoring p^2+i, we can check if p^2+i has a small factor c < p.
    #    If p^2+i = c * q, and c < p, then q = (p^2+i)/c.
    #    If q is prime and q > p, then it's a bad factor.
    #    If q is composite, it might have a prime factor > p.
    #    
    #    This seems to lead back to factorization.
    #    
    #    Let's try a different angle: **Random Search for p**.
    #    The current program tries p=2, 3, 5... This is biased towards small p.
    #    Small p have small p^2. p^2+i is small.
    #    For p=2, p^2=4. 5, 6, 7... 5 is prime > 2. So k=0?
    #    Wait, 5 is prime > 2. So 4+1=5 is not 2-smooth. k=0.
    #    For p=3, p^2=9. 10=2*5 (5>3). k=0.
    #    For p=5, p^2=25. 26=2*13 (13>5). k=0.
    #    For p=7, p^2=49. 50=2*5^2 (all <=7). 51=3*17 (17>7). k=1.
    #    
    #    It seems k is very small for small p.
    #    For large p, p^2 is large.
    #    The condition "all prime factors <= p" is easier to satisfy for large p?
    #    No, the number of primes <= p is ~ p/log(p).
    #    The number of integers <= p^2 that are p-smooth is related to the Dickman function.
    #    
    #    Let's just switch to **Random Sampling of Large Primes**.
    #    Generate random numbers in [10^7, 10^8], check if prime, then check k.
    #    This explores a much larger space than sequential scanning.
    
    # Precompute small primes for trial division in is_p_smooth
    # (Already done in factorize, but is_p_smooth uses pollard_brent directly)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Quick check on small primes (up to 1000) to ensure we have a baseline
    small_primes = sieve(1000)
    for p in small_primes:
        if time.time() > deadline:
            break
        # Check run length for p
        # Use a faster check: just try to find a factor > p
        k = 0
        for i in range(1, 201):
            val = p*p + i
            if not is_p_smooth(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break
    
    # Phase 2: Random search for large primes
    # We will sample random numbers and check if they are prime.
    # If prime, check k.
    
    random.seed(42)
    
    # To avoid duplicate checks, we can keep a set of checked primes, 
    # but for random sampling in a large range, collisions are unlikely.
    
    # Range: 10^7 to 10^8
    # Density of primes is ~ 1/log(10^8) ~ 1/18.
    # So we need to check ~18 random numbers to find 1 prime.
    # Checking primality takes ~log(n) operations.
    # Checking k takes up to 200 * factorization cost.
    
    # Let's estimate time.
    # Factorization of 10^16 number with Pollard-Brent is fast (~ms).
    # So checking one p takes ~ 200 * 1ms = 0.2s.
    # We have 130s. So we can check ~600 primes.
    # To find 600 primes, we need to sample ~600 * 18 = 10800 random numbers.
    # This is very feasible.
    
    # Let's optimize is_p_smooth to be as fast as possible.
    # The current is_p_smooth uses pollard_brent.
    # We can add a quick check for small factors first.
    
    def fast_is_p_smooth(n, p):
        if n == 1:
            return True
        # Check small primes up to 100
        for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if n % q == 0:
                if q > p:
                    return False
                # Divide out q
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                # Recurse on the rest
                return fast_is_p_smooth(n, p)
        
        # If n is prime
        if is_prime(n):
            return n <= p
        
        # Pollard Brent
        d = pollard_brent(n)
        if d == n:
            # Should not happen for composite
            return is_prime(n) and n <= p
        return fast_is_p_smooth(d, p) and fast_is_p_smooth(n // d, p)
    
    # Update check_run to use fast_is_p_smooth
    def check_run_fast(p, max_k=200, deadline=None):
        k = 0
        p2 = p * p
        for i in range(1, max_k + 1):
            if deadline is not None and time.time() > deadline:
                break
            val = p2 + i
            if not fast_is_p_smooth(val, p):
                break
            k += 1
        return k
    
    # Re-run Phase 1 with fast check
    best_k = 0
    best_p = 2
    for p in small_primes:
        if time.time() > deadline:
            break
        k = check_run_fast(p, 200, deadline)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break
    
    # Phase 2: Random Large Primes
    # We will iterate until deadline
    while time.time() < deadline - 5:
        # Pick a random number in [10^7, 10^8]
        # To bias towards larger numbers? Or uniform?
        # Uniform is fine.
        rand_num = random.randint(10_000_000, 100_000_000)
        
        # Check if prime
        if is_prime(rand_num):
            # Check run length
            k = check_run_fast(rand_num, 200, deadline)
            if k > best_k:
                best_k = k
                best_p = rand_num
                if best_k == 200:
                    break
    
    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
