import math
import time

# EVOLVE-BLOCK-START
def _get_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs

def _tau_perp(n):
    divs = _get_divisors(n)
    if len(divs) < 2:
        return 0
    score = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            score += 1
    return score

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    # Candidate 1: Product of first 15 primes
    n1 = 1
    for p in primes[:15]:
        n1 *= p
        
    # Candidate 2: Product of first 14 primes * 2
    n2 = 1
    for p in primes[:14]:
        n2 *= p
    n2 *= 2
    
    # Candidate 3: Product of first 14 primes * 3
    n3 = 1
    for p in primes[:14]:
        n3 *= p
    n3 *= 3

    # Calculate scores
    s1 = _tau_perp(n1)
    s2 = _tau_perp(n2)
    s3 = _tau_perp(n3)
    
    # Select best
    best_n = n1
    best_score = s1
    if s2 > best_score:
        best_n = n2
        best_score = s2
    if s3 > best_score:
        best_n = n3
        best_score = s3
        
    return best_n
# EVOLVE-BLOCK-END
