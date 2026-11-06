"""
Constructs an integer n <= 10^18 maximizing tau_perp(n).

Approach: Model the sorted divisor sequence and use greedy selection of
primes to maximize the density of coprime adjacent pairs. We search
over combinations of primes (and possibly prime powers) to find the
n that maximizes the count of adjacent coprime divisor pairs.
"""

import math
from itertools import combinations

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    divs = [1]
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            new_divs = []
            for d in divs:
                mult = 1
                for e in range(exp + 1):
                    new_divs.append(d * mult)
                    mult *= p
            divs = new_divs
        p += 1 if p == 2 else 2
    if temp > 1:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * temp)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_up_to(limit):
    """Generate primes up to limit using simple sieve."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    # Product of first 15 primes is the baseline
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    base_n = 1
    for p in base_primes:
        base_n *= p
    base_score = _tau_perp(base_n)
    
    best_n = base_n
    best_score = base_score
    
    # Try replacing some primes with larger primes to see if it helps
    # The key insight: we want primes that create many coprime adjacent pairs
    # Let's try various combinations
    
    # First, try the product of first k primes for various k
    all_primes = _primes_up_to(200)
    
    # Try products of first k primes
    for k in range(2, 16):
        if k > len(all_primes):
            break
        prod = 1
        for i in range(k):
            prod *= all_primes[i]
        if prod > 10**18:
            break
        score = _tau_perp(prod)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Try some specific constructions with larger primes
    # The idea: maybe using fewer but larger primes creates better structure
    
    # Try n = product of first 14 primes * some larger prime
    # Or try different prime selections
    
    # Search over subsets of primes to find best combination
    # Focus on products that are <= 10^18
    
    # Try replacing the largest prime with a larger one
    for i in range(10, 15):
        for j in range(i + 1, len(all_primes)):
            if all_primes[j] > 200:
                break
            candidate = base_n // base_primes[i] * all_primes[j]
            if candidate <= 10**18:
                score = _tau_perp(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
    
    # Try adding prime powers: n = p^a * q * r * ... 
    # Sometimes a prime power can create more divisor structure
    
    # Try n = 2^a * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    # (replace 47 with a power of 2)
    for a in range(2, 20):
        candidate = (2 ** a) * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
        if candidate <= 10**18:
            score = _tau_perp(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
        else:
            break
    
    # Try more aggressive searches: different prime sets
    # Use a greedy approach: start with empty set, add primes that maximize score
    
    # For now, return the best found
    return best_n

# EVOLVE-BLOCK-START
def construct():
    # Product of first 15 primes as baseline
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    base_n = 1
    for p in base_primes:
        base_n *= p
    
    best_n = base_n
    best_score = _tau_perp(base_n)
    
    # Try products of first k primes for k = 2..15
    all_primes = _primes_up_to(200)
    for k in range(2, min(16, len(all_primes) + 1)):
        prod = 1
        for i in range(k):
            prod *= all_primes[i]
        if prod > 10**18:
            break
        score = _tau_perp(prod)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Try replacing largest primes with larger ones
    for i in range(8, 15):
        for j in range(i + 1, len(all_primes)):
            candidate = base_n // base_primes[i] * all_primes[j]
            if candidate <= 10**18:
                score = _tau_perp(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
    
    # Try prime power variations
    for a in range(2, 20):
        candidate = (2 ** a) * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
        if candidate <= 10**18:
            score = _tau_perp(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
        else:
            break
    
    return best_n
# EVOLVE-BLOCK-END
