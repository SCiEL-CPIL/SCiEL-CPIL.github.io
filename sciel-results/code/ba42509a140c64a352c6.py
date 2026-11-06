import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11]:
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

    # Pollard's Rho factorization
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n, factors):
        if n == 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        d = pollard_rho(n)
        factorize(d, factors)
        factorize(n // d, factors)

    def is_p_smooth(n, p):
        if n == 1:
            return True
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for q in small_primes:
            if q > p:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        # Use Pollard Rho
        factors = []
        factorize(n, factors)
        for f in factors:
            if f > p:
                return False
        return True

    best_k = 0
    best_p = 2
    
    # Search strategy:
    # 1. Try small primes first? No, larger p gives larger numbers, but smoothness prob is same.
    #    However, for small p, we can check k up to 200 faster? No.
    #    Let's mix small and large.
    
    # First, quickly check primes up to 10000 to find any small k
    for p in range(3, 10000, 2):
        if time.time() > deadline:
            break
        if not is_prime(p):
            continue
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            if is_p_smooth(p*p + k + 1, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            
    # Now random search for larger primes
    # We want to find larger k.
    # The expected number of primes to check for k >= K is ~ 1 / (0.3^K)
    # For K=10, ~ 1/5e-6 = 200,000 primes.
    # For K=15, ~ 1/1e-8 = 100,000,000 primes.
    # We can probably check ~ 10,000 - 50,000 large primes in the remaining time.
    # This might find k=5, 6, 7.
    
    # Let's prioritize checking primes that might have long runs.
    # There's no easy way to predict.
    
    # We can also check specific ranges.
    # Let's just random sample primes in [10^5, 10^9].
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random prime
        # Generate random odd number, check prime
        # To get primes, we can just pick random odd and test.
        # Density is 1/ln(p).
        # For p=10^6, ln ~ 14. So 1 in 14.
        # For p=10^9, ln ~ 21. So 1 in 21.
        
        # Pick a random range to sample from to ensure coverage
        # Lower bound 10^5, Upper bound 10^9
        lower = 100000
        upper = 10**9
        
        # Pick random p
        # To avoid bias towards small p, we can pick log-uniform?
        # Or just uniform in [10^5, 10^9].
        p_candidate = random.randint(lower, upper)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        # Find next prime
        # Miller Rabin is fast.
        while not is_prime(p_candidate):
            p_candidate += 2
            if p_candidate > upper:
                break
                
        if p_candidate > upper:
            continue
            
        # Check smoothness run
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            n = p_candidate * p_candidate + k + 1
            if is_p_smooth(n, p_candidate):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p_candidate
            
    return {'p': best_p, 'k': best_k}
