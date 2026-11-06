import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s margin for safety
    
    # 1. Precompute primes up to 10^6
    limit_sieve = 10**6
    is_prime_sieve = [True] * (limit_sieve + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = False
    for i in range(2, int(limit_sieve**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit_sieve + 1, i):
                is_prime_sieve[j] = False
    
    small_primes = [i for i, prime in enumerate(is_prime_sieve) if prime]
    
    # 2. Miller-Rabin primality test
    def is_prime(n):
        if n < 2: return False
        if n < limit_sieve:
            return is_prime_sieve[n]
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

    # 3. Pollard Rho
    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
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

    # 4. Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p):
        if n == 1:
            return True
        # Try small primes
        for q in small_primes:
            if q > p:
                break
            if q*q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        # n is composite and > 1, and has no small factors <= min(p, 10^6)
        # If p < 10^6, we would have broken out of loop when q > p.
        # But if p < 10^6 and n > 1, then n has a factor > p?
        # If p < 10^6, we tested all primes <= p.
        # If n > 1, then n has a prime factor > p. So return False.
        if p < limit_sieve:
            return False
        
        # If p >= 10^6, n might be prime or composite with large factors.
        # If n is prime, check n <= p.
        # If n is composite, factor it.
        if is_prime(n):
            return n <= p
        
        # Factor n
        factors = []
        def factor(m):
            if m == 1:
                return
            if is_prime(m):
                factors.append(m)
                return
            d = pollard_rho(m)
            factor(d)
            factor(m // d)
        
        factor(n)
        for f in factors:
            if f > p:
                return False
        return True

    # 5. Segmented Sieve Check for Range [p^2+1, p^2+200]
    def check_p(p):
        k = 0
        base = p * p
        # We need to check p^2+1 to p^2+200
        # Initialize remainders
        rem = [base + i + 1 for i in range(200)]
        
        # Sieve with small primes
        # We only need primes up to min(p, sqrt(base+200)) approx p
        # But we only have primes up to 10^6.
        # If p < 10^6, we use primes up to p.
        # If p >= 10^6, we use primes up to 10^6.
        
        max_q = min(p, limit_sieve)
        
        # Iterate through small primes
        for q in small_primes:
            if q > max_q:
                break
            
            # Find first multiple of q in [base+1, base+200]
            # base = p^2
            # We need i such that (base + i) % q == 0
            # i = (-base) % q
            # But we want i in [1, 200]
            start_idx = (-base) % q
            if start_idx == 0:
                start_idx = q
            # Adjust start_idx to be >= 1
            # The numbers are base+1, ..., base+200
            # Index j in rem corresponds to base + j + 1
            # So we need (base + j + 1) % q == 0
            # j + 1 = (-base - 1) % q + k*q
            # j = (-base - 1) % q + k*q - 1
            
            # Let's just find the first index j in [0, 199] such that rem[j] % q == 0
            # rem[j] = base + j + 1
            # (base + 1) % q = r
            # j = (-r) % q
            r = (base + 1) % q
            j = (-r) % q
            if j > 199:
                continue
            
            while j < 200:
                # Divide q out of rem[j]
                while rem[j] % q == 0:
                    rem[j] //= q
                j += q
        
        # Now check remainders
        for i in range(200):
            n = rem[i]
            if n == 1:
                continue
            if is_prime(n):
                if n <= p:
                    continue
                else:
                    break
            else:
                # n is composite
                # If p < limit_sieve, we already know it's not smooth because we divided out all primes <= p
                # and n > 1 implies a factor > p.
                if p < limit_sieve:
                    break
                
                # Check if n is p-smooth
                if not is_p_smooth(n, p):
                    break
        
        # Count k
        # The loop above breaks on the first failure.
        # We need to count how many passed.
        # Let's re-do the counting logic or modify the loop.
        
        # Actually, the loop above doesn't return k directly.
        # Let's restructure check_p to return k.
        
        # Re-implementing check_p properly
        rem = [base + i + 1 for i in range(200)]
        max_q = min(p, limit_sieve)
        
        for q in small_primes:
            if q > max_q:
                break
            r = (base + 1) % q
            j = (-r) % q
            if j > 199:
                continue
            while j < 200:
                while rem[j] % q == 0:
                    rem[j] //= q
                j += q
        
        k = 0
        for i in range(200):
            n = rem[i]
            if n == 1:
                k += 1
                continue
            if is_prime(n):
                if n <= p:
                    k += 1
                else:
                    break
            else:
                if p < limit_sieve:
                    break
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
        return k

    best_p = 2
    best_k = 0
    
    # Check small primes first
    for p in small_primes:
        if time.time() - start_time > time_limit:
            break
        if p == 2:
            # p=2, p^2+1=5. 5>2. k=0.
            continue
        k = check_p(p)
        if k > best_k:
            best_k = k
            best_p = p
            
    # Check random larger primes
    while time.time() - start_time < time_limit:
        p_candidate = random.randint(10**6 + 1, 10**9)
        if p_candidate % 2 == 0:
            p_candidate += 1
        if is_prime(p_candidate):
            k = check_p(p_candidate)
            if k > best_k:
                best_k = k
                best_p = p_candidate

    return {'p': best_p, 'k': best_k}
