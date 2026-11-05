import math
import time
import random

def construct():
    deadline = time.time() + 140.0
    
    # Small primes for trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def is_prime_mr(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        for sp in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == sp: return True
            if n % sp == 0: return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        y, c, m = random.randint(1, n-1), random.randint(1, n-1), 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r-k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1: break
        return g

    def is_p_smooth(n, p):
        # Remove small factors
        for q in small_primes:
            if q > p: break
            while n % q == 0:
                n //= q
        if n == 1: return True
        if is_prime_mr(n):
            return n <= p
        # Pollard's Rho
        # We need to factor n completely to ensure all factors <= p
        # Or just find one factor > p
        # If we find a factor f, check if f <= p.
        # If f > p, return False.
        # If f <= p, divide n by f and recurse.
        
        # To avoid deep recursion and overhead, we can use a stack
        stack = [n]
        while stack:
            curr = stack.pop()
            if curr == 1: continue
            if is_prime_mr(curr):
                if curr > p:
                    return False
                continue
            # Find factor
            f = pollard_brent(curr)
            # f might be composite, but usually small
            # Check if f > p
            # If f > p, it might be a prime factor > p, or composite with factors > p?
            # No, if f is a factor of curr, and f > p, it doesn't immediately mean curr has a prime factor > p.
            # Example: p=10, curr=121=11*11. f=11. 11>10. Return False. Correct.
            # Example: p=10, curr=15=3*5. f=3 or 5. <=10.
            # What if f is composite? e.g. f=121, p=10.
            # pollard_brent returns a non-trivial factor.
            # If f > p, we must check if f has a prime factor > p.
            # But if f is a factor of curr, and f > p, it's possible all prime factors of f are <= p?
            # No, if all prime factors of f are <= p, then f is p-smooth.
            # But we just need to know if curr is p-smooth.
            # If f is p-smooth, we can divide curr by f and check the rest.
            # If f is NOT p-smooth, then curr is NOT p-smooth.
            
            # So:
            # 1. Check if f is p-smooth.
            # 2. If not, return False.
            # 3. If yes, push f and curr//f onto stack.
            
            # But checking if f is p-smooth is the same problem.
            # However, f < curr. So recursion terminates.
            
            # Optimization: If f > p, we can just push f and curr//f.
            # The stack will handle the primality/smoothness check of f.
            
            stack.append(f)
            stack.append(curr // f)
        return True

    best_k = 0
    best_p = 2
    
    # Check small primes explicitly to ensure baseline
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        k = 0
        for j in range(1, 201):
            if is_p_smooth(p*p + j, p):
                k = j
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Random search for larger primes
    random.seed(42)
    checked = set()
    
    # We want to find a long run.
    # Try primes in range [10^3, 10^7]
    # Sample ~100,000 candidates
    
    while time.time() < deadline:
        # Pick a random number
        # Bias towards smaller p? 
        # Smaller p is faster to check.
        # Let's try uniform in [1000, 1000000]
        x = random.randint(1000, 1000000)
        if x in checked:
            continue
        checked.add(x)
        if not is_prime_mr(x):
            continue
        p = x
        k = 0
        for j in range(1, 201):
            if is_p_smooth(p*p + j, p):
                k = j
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 20:
                break # Unlikely, but good to have
        
    return {'p': best_p, 'k': best_k}
