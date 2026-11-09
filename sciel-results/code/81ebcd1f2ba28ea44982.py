import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return n == 2 or n == 3
    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
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

# ---------- Pollard's Rho with Brent's improvement ----------
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        q = 1
        r = 1
        m = 128
        ys = y
        while d == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and d == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                d = math.gcd(q, n)
                k += m
            r *= 2
        if d != n:
            return d
        d = 1
        while d == 1:
            ys = (ys * ys + c) % n
            d = math.gcd(abs(x - ys), n)
        if d != n:
            return d

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, limit + 1, i):
                is_p[j] = 0
    return [i for i in range(2, limit + 1) if is_p[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    
    # Trial division by small primes
    for q in small_primes:
        if q > p or q * q > n:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
    
    if n == 1:
        return True
    
    # If remaining n is <= p, it's smooth (since any prime factor of n is <= n <= p)
    if n <= p:
        return True
    
    # n > p. Check if n is prime.
    if is_prime(n):
        # n is a prime factor > p
        return False
    
    # n is composite and > p.
    # Find a factor
    d = pollard_rho(n)
    # Recursively check factors
    return is_p_smooth(d, p, small_primes) and is_p_smooth(n // d, p, small_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to 10^7 for trial division and as candidates for p
    SIEVE_LIMIT = 10**7
    primes = sieve_primes(SIEVE_LIMIT)
    
    # We will test primes p from the list.
    # To maximize k, larger p is generally better.
    # However, we should probably test a wide range.
    # Let's test all primes in the list.
    
    # Optimization: Pre-filtering
    # For a given p, if p^2 + 1 is not p-smooth, k=0.
    
    # We iterate through primes.
    # To save time, we can skip very small primes where k is likely small.
    # But we don't know the distribution.
    
    # Let's iterate from the end (largest p) to the beginning?
    # If we find a large k early, we can stop? No, we want max k.
    # But checking large p is slower.
    # Let's iterate from small to large, but with a time check.
    
    # Actually, let's iterate from the middle or large side?
    # The probability of smoothness increases with p.
    # So larger p is more likely to have long runs.
    # Let's iterate from the largest primes downwards, but we need to be careful with time.
    # Or just iterate all, and if time runs out, return best.
    
    # To speed up, we can check p in reverse order (largest first).
    # If we find a k, say 10, we still need to check if a larger k exists.
    # But if we find k=50, it's very good.
    
    # Let's just iterate all primes up to SIEVE_LIMIT.
    # 10^7 has ~660k primes.
    # For each p, we check k=1..200.
    # Most will fail at k=1.
    # Check for k=1: is_p_smooth(p*p+1, p, primes)
    # This involves trial division up to min(p, 10^7).
    # For p=10^7, trial division is ~660k steps.
    # 660k primes * 660k steps = 4e11 steps. Too slow.
    
    # We need a faster pre-filter.
    # For p^2+1, we can check divisibility by small primes first.
    # If p^2+1 is not divisible by any small prime, it might be prime or have large factors.
    # If it's prime, it fails (unless p=2).
    # So if p^2+1 is prime, k=0.
    
    # Let's modify the loop:
    # 1. Check if p^2+1 is p-smooth.
    #    - Trial divide p^2+1 by primes up to 1000.
    #    - If remainder is 1, ok.
    #    - If remainder <= p, ok.
    #    - If remainder > p, check if prime. If prime, fail.
    #    - If composite, pollard rho.
    
    # This is still heavy for all primes.
    # We should limit the number of primes we check or optimize the smoothness check.
    
    # Strategy:
    # 1. Generate primes up to 10^7.
    # 2. Iterate through them.
    # 3. For each p, check k.
    # 4. If time < 10s remaining, stop.
    
    # To make it faster, we can use a smaller sieve for trial division in is_p_smooth?
    # No, we need to divide by all primes up to p to be sure.
    # But if p is large, and n is p^2+1, the smallest factor of n (if composite) is likely small?
    # No, p^2+1 can be prime.
    
    # Let's try to check primes in a specific range first.
    # Maybe p between 10^5 and 10^7?
    
    # Let's iterate through the primes list.
    # To ensure we don't timeout, we check time every 100 primes.
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Skip very small primes? They are fast to check but unlikely to have large k.
        # But we should check them to be safe.
        
        k = 0
        p_sq = p * p
        
        # Check up to 200
        while k < 200:
            if time.time() > deadline:
                break
            
            n = p_sq + k + 1
            if not is_p_smooth(n, p, primes):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
        # If we found a very high k, we might want to continue searching for even higher?
        # Yes.
        
    return {'p': best_p, 'k': best_k}

# Test with small values
# p=2: 4+1=5 (fail). k=0.
# p=3: 9+1=10 (2*5, 5>3 fail). k=0.
# p=5: 25+1=26 (2*13, 13>5 fail). k=0.
# p=7: 49+1=50 (2*5*5, all <=7). 49+2=51 (3*17, 17>7 fail). k=1.
# p=11: 121+1=122 (2*61, 61>11 fail). k=0.
# p=13: 169+1=170 (2*5*17, 17>13 fail). k=0.
# p=17: 289+1=290 (2*5*29, 29>17 fail). k=0.
# p=19: 361+1=362 (2*181, 181>19 fail). k=0.
# p=23: 529+1=530 (2*5*53, 53>23 fail). k=0.
# p=29: 841+1=842 (2*421, 421>29 fail). k=0.
# p=31: 961+1=962 (2*13*37, 37>31 fail). k=0.
# p=37: 1369+1=1370 (2*5*137, 137>37 fail). k=0.
# p=41: 1681+1=1682 (2*29*29, 29<=41 ok). 1681+2=1683 (3*3*37*5, 37<=41 ok). 1681+3=1684 (2*2*421, 421>41 fail). k=2.
# p=43: 1849+1=1850 (2*5*5*37, ok). 1849+2=1851 (3*617, 617>43 fail). k=1.
# p=47: 2209+1=2210 (2*5*13*17, ok). 2209+2=2211 (3*11*67, 67>47 fail). k=1.
# p=53: 2809+1=2810 (2*5*281, 281>53 fail). k=0.
# p=59: 3481+1=3482 (2*1741, 1741>59 fail). k=0.
# p=61: 3721+1=3722 (2*1861, 1861>61 fail). k=0.
# p=67: 4489+1=4490 (2*5*449, 449>67 fail). k=0.
# p=71: 5041+1=5042 (2*2521, 2521>71 fail). k=0.
# p=73: 5329+1=5330 (2*5*26*41? 5330/10=533=13*41. 2*5*13*41. All <=73. OK).
# 5329+2=5331 (3*1777, 1777>73 fail). k=1.
# p=79: 6241+1=6242 (2*3121, 3121>79 fail). k=0.
# p=83: 6889+1=6890 (2*5*13*53, ok). 6889+2=6891 (3*2297, 2297>83 fail). k=1.
# p=89: 7921+1=7922 (2*3961, 3961>89 fail). k=0.
# p=97: 9409+1=9410 (2*5*47*20? 9410/10=941 prime? 941>97 fail). k=0.

# It seems k is small for small p.
# We need larger p.

# The current code should work.
# One optimization: if p is very large, p^2+1 is very large.
# is_p_smooth will do trial division up to p.
# If p is 10^7, that's 660k divisions.
# If we check 1000 primes, that's 660M divisions.
# In Python, 660M divisions might take ~30-60s.
# So we can check ~1000-2000 primes in the time budget.
# We have 660k primes.
# We should probably check the largest primes first?
# Or a random sample?
# The probability of a long run is higher for larger p.
# Let's iterate from the end of the primes list (largest p) to the beginning.
# This way, if we find a good k early, we can stop?
# No, we want to maximize k.
# But if we find k=10, and we check smaller p, we might find k=11?
# Unlikely, but possible.
# However, checking large p is slower.
# Let's just iterate from largest to smallest, and break if time runs out.
# This prioritizes the most likely candidates for long runs.

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_k = 0
    best_p = 2
    
    SIEVE_LIMIT = 10**7
    primes = sieve_primes(SIEVE_LIMIT)
    
    # Iterate from largest to smallest
    for p in reversed(primes):
        if time.time() > deadline:
            break
        
        k = 0
        p_sq = p * p
        
        while k < 200:
            if time.time() > deadline:
                break
            
            n = p_sq + k + 1
            if not is_p_smooth(n, p, primes):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
