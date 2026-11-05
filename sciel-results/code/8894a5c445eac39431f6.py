import math
import time
from itertools import product as iproduct

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Primes up to 53
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    def get_divisors(n):
        """Get all divisors of n using its prime factorization."""
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

    candidates = []
    
    # Candidate 1: Product of first 15 primes (Square-free)
    n_15 = 1
    for p in primes[:15]:
        n_15 *= p
    if n_15 <= 10**18:
        candidates.append(n_15)
        
    # Candidate 2: Product of first 14 primes, with 2 squared
    n_14_sq2 = 2 * 1
    for p in primes[1:14]:
        n_14_sq2 *= p
    n_14_sq2 *= 2
    if n_14_sq2 <= 10**18:
        candidates.append(n_14_sq2)
        
    # Candidate 3: Product of first 14 primes, with 3 squared
    n_14_sq3 = 2 * 1
    for p in primes[1:14]:
        n_14_sq3 *= p
    n_14_sq3 *= 3
    if n_14_sq3 <= 10**18:
        candidates.append(n_14_sq3)
        
    # Candidate 4: Product of first 13 primes, with 2 and 3 squared
    n_13_sq23 = 4 * 9 * 1
    for p in primes[2:13]:
        n_13_sq23 *= p
    if n_13_sq23 <= 10**18:
        candidates.append(n_13_sq23)
        
    # Candidate 5: Product of first 13 primes, with 2 cubed
    n_13_cub2 = 8 * 1
    for p in primes[1:13]:
        n_13_cub2 *= p
    if n_13_cub2 <= 10**18:
        candidates.append(n_13_cub2)

    # Candidate 6: Product of first 13 primes, with 3 cubed
    n_13_cub3 = 2 * 27 * 1
    for p in primes[2:13]:
        n_13_cub3 *= p
    if n_13_cub3 <= 10**18:
        candidates.append(n_13_cub3)

    # Now do a smarter search: try various exponent patterns
    # Key insight: we want many divisors AND many coprime adjacent pairs
    # Let's try numbers of form 2^a * 3^b * 5^c * ... with small exponents
    
    # Generate candidates by trying different exponent combinations
    # Focus on numbers with many prime factors
    
    # Try: 2^a * 3^b * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    # with various (a,b) where a,b in {1,2,3,4}
    
    base_primes = primes[2:13]  # 5,7,11,13,17,19,23,29,31
    base_product = 1
    for p in base_primes:
        base_product *= p
    
    for a in range(1, 6):
        for b in range(1, 6):
            n = (2**a) * (3**b) * base_product
            if n <= 10**18:
                candidates.append(n)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Try: 2^a * 3^b * 5^c * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    base_primes2 = primes[3:12]  # 7,11,13,17,19,23,29,31
    base_product2 = 1
    for p in base_primes2:
        base_product2 *= p
    
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                n = (2**a) * (3**b) * (5**c) * base_product2
                if n <= 10**18:
                    candidates.append(n)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Try: 2^a * 3^b * 5^c * 7^d * 11 * 13 * 17 * 19 * 23 * 29
    base_primes3 = primes[4:11]  # 11,13,17,19,23,29
    base_product3 = 1
    for p in base_primes3:
        base_product3 *= p
    
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                for d in range(1, 5):
                    n = (2**a) * (3**b) * (5**c) * (7**d) * base_product3
                    if n <= 10**18:
                        candidates.append(n)
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Also try some highly composite-like structures
    # 2^4 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    n_hc1 = (2**4) * (3**3) * (5**2) * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    if n_hc1 <= 10**18:
        candidates.append(n_hc1)
    
    # 2^5 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    n_hc2 = (2**5) * (3**3) * (5**2) * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    if n_hc2 <= 10**18:
        candidates.append(n_hc2)
    
    # 2^4 * 3^4 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    n_hc3 = (2**4) * (3**4) * (5**2) * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    if n_hc3 <= 10**18:
        candidates.append(n_hc3)

    # Remove duplicates
    candidates = list(set(candidates))
    
    best_n = candidates[0]
    best_score = -1
    
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n
