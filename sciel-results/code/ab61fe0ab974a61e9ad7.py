import time
import math

def construct():
    deadline = time.time() + 110
    
    def is_prime_mr(n):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0 or n % 3 == 0: return False
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
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
    
    def is_p_smooth(n, p):
        if n <= 1: return True
        orig = n
        # Trial divide by small primes
        for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % d == 0:
                if d > p:
                    return False
                n //= d
        if n == 1: return True
        if n > p:
            # Check if n is prime
            if is_prime_mr(n):
                return n <= p
            # n is composite, try to factor
            # For now, assume not smooth if n > p and we can't factor quickly
            # Try a few more divisions
            d = 101
            while d * d <= n and d <= p:
                while n % d == 0:
                    n //= d
                d += 2
            if n > 1 and n > p:
                return False
            return True
        return True
    
    best_k = 0
    best_p = 2
    
    # Search primes
    p = 2
    while p <= 10**6 and time.time() < deadline:
        if is_prime_mr(p):
            k = 0
            base = p * p
            while k < 200 and time.time() < deadline:
                n = base + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        p += 1 if p == 2 else 2
    
    return {'p': best_p, 'k': best_k}
