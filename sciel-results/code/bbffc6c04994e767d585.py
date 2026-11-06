import time
import math
import numpy as np

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # 1. Sieve primes up to B = 10^6
    B = 10**6
    sieve = np.ones(B + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(math.isqrt(B)) + 1):
        if sieve[i]:
            sieve[i*i:B+1:i] = False
    primes = np.array(np.nonzero(sieve))[0]
    primes_list = primes.tolist()
    num_primes = len(primes_list)
    
    # 2. Miller-Rabin for primality test
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0 or n % 11 == 0 or n % 13 == 0 or n % 17 == 0 or n % 19 == 0 or n % 23 == 0 or n % 29 == 0 or n % 31 == 0 or n % 37 == 0:
            return False
        # Deterministic Miller-Rabin for n < 3,474,749,660,383
        # Our n is <= 10^18 + 200, so we need more bases or a different method.
        # For n < 2^64, the set of bases {2, 3, 5, 7, 11, 13, 17} is not enough.
        # Bases {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} work for n < 3.1e14.
        # For n < 2^64, we can use {2, 325, 9375, 28178, 450775, 9780504, 1795265022}.
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases for 64-bit integers
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard-Brent for finding a factor
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = int(np.random.randint(2, n))
            c = int(np.random.randint(1, n))
            m = 128
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
            # Retry with new parameters if g == n

    # 4. Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if p <= 0:
            return n == 1
        
        # If p is small, we can just trial divide by primes <= p
        # Find index of last prime <= p
        # Binary search in primes_list
        idx = np.searchsorted(primes, p, side='right')
        
        # Divide by small primes
        for i in range(idx):
            q = primes_list[i]
            if q * q > n:
                break
            while n % q == 0:
                n //= q
            if n == 1:
                return True
            if time.time() - start_time > timeout:
                return False # Timeout safety, assume fail to stop loop
        
        if n == 1:
            return True
        
        # If p >= B, we've divided by all primes <= B.
        # If p < B, we've divided by all primes <= p.
        # In either case, if n > 1:
        # If p >= B, n has no factors <= B. Since B < p, n might have factors in (B, p].
        # If p < B, n has no factors <= p. So any factor of n is > p.
        
        if p < B:
            # We divided by all primes <= p.
            # If n > 1, it has a prime factor > p.
            return False
        
        # p >= B
        # n has no prime factors <= B.
        # We need to check if n has any prime factor > p.
        # If n is prime, check if n <= p.
        # If n is composite, find smallest prime factor q. If q > p, fail.
        
        while n > 1:
            if is_prime(n):
                if n <= p:
                    return True
                else:
                    return False
            
            # Find a factor of n
            q = pollard_brent(n)
            
            # q is a factor. It might be composite? Pollard-Brent usually finds a non-trivial factor.
            # We should check if q is prime. If not, factor q further?
            # Actually, if q is composite, it has a prime factor <= sqrt(q) <= sqrt(n).
            # We just need to know if ALL prime factors of n are <= p.
            # So we can just divide n by q and repeat?
            # No, if q is composite, say q = a*b, and a > p, then fail.
            # So we need the prime factors.
            # Better: find smallest prime factor.
            # Pollard-Brent doesn't guarantee smallest.
            # But we can just factor n completely into primes?
            # Or just check: does n have a prime factor > p?
            # If we find a factor q, we can check if q is prime.
            # If q is prime:
            #   If q > p: return False
            #   Else: n //= q, continue
            # If q is composite:
            #   It has a prime factor r <= sqrt(q).
            #   If sqrt(q) > p, then r might be > p.
            #   This is getting complicated.
            
            # Simpler: Since n has no factors <= B (10^6), and p >= B.
            # If n is composite, its smallest prime factor q is > B.
            # If q > p, return False.
            # If q <= p, divide n by q and repeat.
            
            # Let's get a factor q.
            # If q is prime:
            if is_prime(q):
                if q > p:
                    return False
                while n % q == 0:
                    n //= q
            else:
                # q is composite. It has a prime factor r.
                # We need to find r.
                # We can recursively factor q?
                # Or just find any prime factor of q.
                # Since q < n, and n has no factors <= B, q has no factors <= B.
                # So q is a product of primes > B.
                # We can find a prime factor of q using the same logic.
                # Let's just use a helper to get smallest prime factor of a number known to be > B-smooth.
                
                def smallest_prime_factor_large(m):
                    # m > 1, no factors <= B
                    if is_prime(m):
                        return m
                    f = pollard_brent(m)
                    # f is a factor.
                    # If f is prime, return f? No, f might not be the smallest.
                    # But we just need *a* prime factor.
                    # If we find a prime factor r, we can check if r > p.
                    # If r <= p, we divide m by r and repeat.
                    # This implies we just need to extract prime factors one by one.
                    
                    # Let's refactor:
                    # We have n.
                    # Loop:
                    #   if n == 1: return True
                    #   if is_prime(n):
                    #       if n <= p: return True
                    #       else: return False
                    #   f = pollard_brent(n)
                    #   if is_prime(f):
                    #       if f > p: return False
                    #       n //= f
                    #   else:
                    #       # f is composite.
                    #       # We need to factor f.
                    #       # We can just push f onto a stack?
                    #       # Or just find a prime factor of f.
                    #       # Let's find a prime factor of f.
                    #       # This is getting deep.
                    #       # Alternative:
                    #       # Since f < n, and we are in a loop, we can just treat f as the new n?
                    #       # No, we need to clear n.
                    #       # Let's just find *any* prime factor of n.
                    #       # If we find a composite factor, we can just keep factoring it until we get primes.
                    #       # Let's make a stack of numbers to factor.
                    pass
                
                # Let's implement a stack-based factorization for the "large" part.
                stack = [n]
                while stack:
                    m = stack.pop()
                    if m == 1:
                        continue
                    if is_prime(m):
                        if m > p:
                            return False
                    else:
                        f = pollard_brent(m)
                        # f is a non-trivial factor
                        # It might be prime or composite
                        # We push m//f and f
                        stack.append(m // f)
                        stack.append(f)
                        if time.time() - start_time > timeout:
                            return False
                return True

    # 5. Main search loop
    best_k = 0
    best_p = 2
    
    # We can iterate p.
    # To be efficient, we can iterate primes p.
    # But p can be up to 10^9.
    # We have primes up to 10^6.
    # For p > 10^6, we need to generate primes or just test odd numbers?
    # Testing odd numbers for primality is expensive.
    # But we can just iterate p from 2 to some limit?
    # Or iterate p in chunks?
    
    # Let's iterate p from 2 upwards.
    # We need to check if p is prime.
    # For p <= 10^6, we have the sieve.
    # For p > 10^6, we use Miller-Rabin.
    
    # To speed up, we can skip p if p^2+1 is not p-smooth.
    
    p = 2
    while True:
        if time.time() - start_time > timeout:
            break
        
        # Check if p is prime
        if p <= B:
            is_p_prime = sieve[p]
        else:
            is_p_prime = is_prime(p)
            
        if is_p_prime:
            # Find max k for this p
            k = 0
            while k < 200:
                n = p * p + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                # Optional: if best_k == 200, we can stop?
                # The task says 1 <= k <= 200.
                # If we find 200, it's the max possible.
                if best_k == 200:
                    break
        
        p += 1 if p == 2 else 2
        
    return {'p': best_p, 'k': best_k}
