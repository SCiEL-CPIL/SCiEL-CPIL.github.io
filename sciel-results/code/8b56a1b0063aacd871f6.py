import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 118  # 2s margin

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases for deterministic test for 64-bit integers
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            if a % n == 0:
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

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
        while True:
            y = random.randint(1, n - 1)
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

    # Sieve primes up to a limit
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Small primes for trial division
    small_primes_list = sieve(1000)

    # Check if n is p-smooth
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return True
        
        # Trial divide by small primes
        for sp in small_primes_list:
            if sp > p:
                break
            if sp * sp > n:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return True
                if n <= p:
                    return True
                # If n is now prime, check if <= p
                if is_prime(n):
                    return n <= p
        
        if n == 1:
            return True
        if n <= p:
            return True
        
        # n is composite and has no small factors
        # Check if n is prime
        if is_prime(n):
            return n <= p
        
        # n is composite. Find a factor.
        # If any prime factor > p, return False.
        # We can use Pollard Rho to find a factor d.
        # If d is prime and d > p, return False.
        # If d is prime and d <= p, check n/d.
        # If d is composite, check d and n/d.
        
        def check_smooth_recursive(m, p):
            if m <= 1:
                return True
            if m <= p:
                return True
            if is_prime(m):
                return m <= p
            
            # Try to find a factor
            d = pollard_rho(m)
            # d is a non-trivial factor
            if d == m:
                # Failed to find factor? Should not happen for composite
                return False
            
            # Check d and m//d
            if not check_smooth_recursive(d, p):
                return False
            if not check_smooth_recursive(m // d, p):
                return False
            return True

        return check_smooth_recursive(n, p)

    best_k = 0
    best_p = 2

    # Search primes
    # We will search up to 10^8 if time permits
    limits = [10**6, 10**7, 5*10**7, 10**8]
    prev_limit = 100
    
    for limit in limits:
        if time.time() - start_time > deadline - 10:
            break
            
        primes = sieve(limit)
        
        for p in primes:
            if p <= prev_limit:
                continue
            if time.time() - start_time > deadline - 2:
                break
                
            # Quick check for k=1
            if not is_p_smooth(p * p + 1, p):
                continue
            
            k = 1
            while k < 200:
                if time.time() - start_time > deadline - 1:
                    break
                n = p * p + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
        
        prev_limit = limit

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
