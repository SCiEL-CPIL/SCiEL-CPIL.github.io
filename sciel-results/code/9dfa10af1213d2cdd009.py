import time
import random
from math import gcd

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic witnesses for 64-bit integers
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
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
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def sieve(n):
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, n + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def is_p_smooth(n, p, small_primes):
    if n == 1:
        return True
    # Trial division by small primes
    for q in small_primes:
        if q > p:
            # If we have tried all primes up to p (or limit), 
            # and n is still > 1, any prime factor of n must be > p?
            # No, small_primes is a fixed list.
            # If q > p, we break? 
            # If we break, we haven't divided by primes between current q and p.
            # But if small_primes covers up to 10^6, and p > 10^6, 
            # we divide by all small_primes.
            # If p < 10^6, we should stop at p.
            # Let's handle this: if q > p, we can stop trial division.
            # But then remaining n might have factors <= p that we didn't check?
            # No, small_primes are sorted. If q > p, all subsequent are > p.
            # So we have checked all primes <= p (that are in small_primes).
            # If p < max(small_primes), we missed primes in (max_checked, p]?
            # No, small_primes contains ALL primes up to limit.
            # So if q > p, we have checked all primes <= p.
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    # n is composite
    # Use Pollard Brent
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > p:
                return False
        else:
            d = pollard_brent(m)
            if d == m:
                # Failure? Retry or handle
                # pollard_brent should not return m usually
                # If it does, retry
                continue 
            stack.append(d)
            stack.append(m // d)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Sieve primes up to 10^7 for candidate p
    # And we need small_primes for trial division in is_p_smooth
    # Let's sieve up to 10^6 for small_primes (fast)
    # And sieve up to 10^7 for candidates?
    # Actually, if we sieve up to 10^7, we get both.
    limit = 10**7
    primes = sieve(limit)
    small_primes = [p for p in primes if p < 10**6] # Use subset for trial division to keep it fast?
    # Actually, dividing by primes up to 10^6 is ~78k divisions.
    # For each n, this is fast.
    
    best_k = 0
    best_p = 2
    
    # We can search primes in any order. Smaller p might have longer runs?
    # Or larger p?
    # Let's iterate through primes.
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Check p^2 + 1
        n = p * p + 1
        if not is_p_smooth(n, p, small_primes):
            continue
        
        k = 1
        # Check further j
        while k < 200:
            if time.time() > deadline:
                break
            n = p * p + k + 1
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            # print(f"Found p={p}, k={k}")

    return {'p': best_p, 'k': best_k}
