import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_divisors(n):
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if temp == 1:
                break
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        return sorted(divs)

    def calc_score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Systematic search over exponent combinations
    # Key insight: more distinct primes with exponent 1 tend to create more coprime adjacent pairs
    # Let's explore: which primes to include, and what exponents to use for small primes
    
    best_n = 2
    best_score = 0
    
    # Strategy: For each subset of primes (up to ~14 primes to stay under 10^18),
    # try different exponent assignments for the first few primes
    
    # First, find the maximum number of distinct primes we can use
    # Product of first k primes:
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43 = ~1.3e17
    # *47 = ~6.1e18 > 10^18
    # So max 14 distinct primes (up to 43)
    
    # For 14 primes (2..43), try various exponent patterns for 2 and 3
    base_primes_14 = primes[:14]  # 2,3,5,7,11,13,17,19,23,29,31,37,41,43
    base_primes_13 = primes[:13]  # 2,3,5,7,11,13,17,19,23,29,31,37,41
    
    # Try different exponent combinations for first two primes
    exponent_options = [
        (1, 1),  # square-free
        (2, 1),
        (1, 2),
        (3, 1),
        (1, 3),
        (2, 2),
        (4, 1),
        (1, 4),
        (3, 2),
        (2, 3),
        (5, 1),
        (1, 5),
        (4, 2),
        (2, 4),
        (3, 3),
        (6, 1),
        (1, 6),
    ]
    
    candidates = []
    
    # Try 14 primes with various exponents
    for e2, e3 in exponent_options:
        n = (2 ** e2) * (3 ** e3)
        for p in base_primes_14[2:]:
            n *= p
        if 2 <= n <= 10**18:
            candidates.append(n)
    
    # Try 13 primes with more aggressive exponents
    for e2 in range(1, 8):
        for e3 in range(1, 8):
            if e2 + e3 > 10:  # limit total
                continue
            n = (2 ** e2) * (3 ** e3)
            for p in base_primes_13[2:]:
                n *= p
            if 2 <= n <= 10**18:
                candidates.append(n)
    
    # Also try 12 primes with even more aggressive exponents
    base_primes_12 = primes[:12]
    for e2 in range(1, 10):
        for e3 in range(1, 10):
            if e2 + e3 > 14:
                continue
            n = (2 ** e2) * (3 ** e3)
            for p in base_primes_12[2:]:
                n *= p
            if 2 <= n <= 10**18:
                candidates.append(n)
    
    # Remove duplicates and evaluate
    candidates = list(set(candidates))
    
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n
