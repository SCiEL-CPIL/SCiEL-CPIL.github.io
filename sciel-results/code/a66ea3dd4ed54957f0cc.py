import math
import time

def construct():
    global small_primes_global, deadline_global
    start_time = time.time()
    deadline_global = start_time + 140.0
    
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    small_primes_global = sieve(100)
    
    def is_prime_fast(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = 2, 1, 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for i in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for i in range(min(m, r - k)):
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
        return g

    def is_p_smooth(n, p):
        if n <= p:
            return True
        for q in small_primes_global:
            if q > p:
                break
            while n % q == 0:
                n //= q
        if n == 1:
            return True
        if n <= p:
            return True
        if is_prime_fast(n):
            return False
        factor = pollard_brent(n)
        if factor == -1:
            return is_prime_fast(n)
        return is_p_smooth(factor, p) and is_p_smooth(n // factor, p)

    def verify_run(p, max_k):
        k = 0
        p2 = p * p
        for i in range(1, max_k + 1):
            if time.time() > deadline_global - 5:
                break
            if not is_p_smooth(p2 + i, p):
                break
            k += 1
        return k

    # Generate candidates
    candidate_limit = 100000
    candidates = sieve(candidate_limit)
    
    scored = []
    for p in candidates:
        if time.time() > deadline_global - 30:
            break
        score = 0
        p2 = p * p
        for i in range(1, 201):
            x = p2 + i
            for q in small_primes_global:
                if q * q > x:
                    break
                while x % q == 0:
                    x //= q
            if x == 1:
                score += 1
            elif x <= p:
                score += 1
            else:
                if is_prime_fast(x):
                    pass
                else:
                    # Heuristic: if x is composite and > p, it might be smooth
                    # but we don't count it to be safe?
                    # Or we can count it with lower weight?
                    # Let's not count it.
                    pass
        scored.append((score, p))
        
    scored.sort(reverse=True)
    
    best_k = 0
    best_p = 0
    for score, p in scored[:100]:
        if time.time() > deadline_global - 10:
            break
        k = verify_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
                
    if best_p == 0:
        # Fallback
        best_p = 2
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
