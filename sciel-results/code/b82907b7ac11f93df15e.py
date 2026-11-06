import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Precompute primes up to 10^7 for trial division / sieving
    P_MAX = 10000000
    is_prime_sieve = bytearray(b'\x01') * (P_MAX + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, P_MAX + 1) if is_prime_sieve[i]]
    primes_set = set(primes)

    # Deterministic Miller-Rabin for n < 3.2 * 10^18
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    # Pollard-Brent factorization
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
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g != 1:
                        break
            if g != n:
                return g

    def factor(n, factors=None):
        if factors is None:
            factors = []
        if n <= 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                factor(p, factors)
                factor(n // p, factors)
                return factors
        d = pollard_brent(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    def check_range(p):
        """
        Check if p^2+1...p^2+200 are p-smooth.
        Returns max k.
        """
        if time.time() - start_time > DEADLINE:
            return 0
            
        p2 = p * p
        # We will sieve the range [p2+1, p2+200] with primes <= min(p, P_MAX)
        # rem[j] stores the remaining factor of p2+j after dividing by small primes
        rem = [p2 + j for j in range(1, 201)]
        
        # Sieve with small primes
        # We only need primes up to min(p, P_MAX)
        limit = min(p, P_MAX)
        
        # Optimization: Only iterate primes that are <= limit
        # We can break early if prime > limit
        for q in primes:
            if q > limit:
                break
            
            # Find first index j in 1..200 such that p2+j is divisible by q
            # p2 + j = k*q => j = k*q - p2
            # j >= 1 => k*q >= p2 + 1 => k >= ceil((p2+1)/q)
            k_start = (p2 + 1 + q - 1) // q
            j_start = k_start * q - p2
            
            # Iterate through multiples
            for j in range(j_start, 201, q):
                while rem[j] % q == 0:
                    rem[j] //= q
        
        # Now check remainders
        k_max = 0
        for j in range(1, 201):
            r = rem[j]
            if r == 1:
                k_max = j
                continue
            
            # Factor the remainder
            # If remainder has any factor > p, it's not p-smooth
            facs = factor(r)
            is_smooth = True
            for f in facs:
                if f > p:
                    is_smooth = False
                    break
            
            if is_smooth:
                k_max = j
            else:
                break
        
        return k_max

    best_k = 1
    best_p = 2
    
    # Search strategy:
    # 1. Try some small primes to establish a baseline
    # 2. Try random primes in [10^4, 10^9]
    
    # Small primes
    for p in primes[:100]:
        if time.time() - start_time > DEADLINE:
            break
        k = check_range(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Random search for larger primes
    # We need to generate random primes up to 10^9
    # We can pick a random number and find the next prime
    
    def next_prime(n):
        if n <= 2:
            return 2
        if n % 2 == 0:
            n += 1
        else:
            n += 2
        while not is_prime_mr(n):
            n += 2
        return n

    random.seed(42)
    # Search in log-uniform distribution of p
    for _ in range(5000):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Pick random p in [10^4, 10^9]
        # log10(p) in [4, 9]
        log_p = random.uniform(4, 9)
        p = int(10 ** log_p)
        p = next_prime(p)
        
        if p > 10**9:
            continue
            
        k = check_range(p)
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good k, we might want to search nearby?
            # Or just continue random.

    return {'p': best_p, 'k': best_k}
