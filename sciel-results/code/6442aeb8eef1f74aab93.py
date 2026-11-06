"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Structural approach: bounded search over prime exponent vectors.
We explore:
  - Square-free numbers with 14-15 distinct primes (varying which primes)
  - Numbers with 13-14 primes where one or two primes have exponent 2
We compute tau_perp for each candidate and return the best.
"""
import math
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    # Get all divisors
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort()
    
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j+1]) == 1:
            count += 1
    return count

def _tau_perp_from_exponents(primes, exponents):
    """Compute tau_perp more efficiently by generating divisors from factorization."""
    # Generate all divisors from prime factorization
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    divisors.sort()
    
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Base primes (first 20 primes)
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    best_n = 1
    best_score = 0
    
    # Phase 1: Square-free with 15 primes (the primorial)
    # Try the standard first-15-primes
    p15 = base_primes[:15]
    n15 = 1
    for p in p15:
        n15 *= p
    if n15 <= 10**18:
        score = _tau_perp_from_exponents(p15, [1]*15)
        if score > best_score:
            best_score = score
            best_n = n15
    
    # Phase 2: Try swapping the largest 1-3 primes with larger ones
    # Keep the product <= 10^18
    for num_swaps in range(1, 4):
        if time.time() > deadline:
            break
        # Try replacing the last `num_swaps` primes with other primes
        # The product of first 15 primes is ~6.15e17, so we have room
        # to increase the largest primes
        fixed_primes = base_primes[:15 - num_swaps]
        fixed_prod = 1
        for p in fixed_primes:
            fixed_prod *= p
        
        # We can choose `num_swaps` primes from the larger primes
        # such that fixed_prod * (product of chosen) <= 10^18
        max_allowed = 10**18 // fixed_prod
        # Candidate larger primes (from index 14 onward, but we already used up to 47)
        # Actually we should try replacing with primes > 47 or different combinations
        larger_primes = base_primes[14:]  # 47, 53, 59, 61, 67, 71
        
        if num_swaps == 1:
            for p in larger_primes:
                if fixed_prod * p <= 10**18:
                    primes_list = fixed_primes + [p]
                    n = fixed_prod * p
                    score = _tau_perp_from_exponents(primes_list, [1]*len(primes_list))
                    if score > best_score:
                        best_score = score
                        best_n = n
        elif num_swaps == 2:
            for p1, p2 in itertools.combinations(larger_primes, 2):
                if fixed_prod * p1 * p2 <= 10**18:
                    primes_list = fixed_primes + [p1, p2]
                    n = fixed_prod * p1 * p2
                    score = _tau_perp_from_exponents(primes_list, [1]*len(primes_list))
                    if score > best_score:
                        best_score = score
                        best_n = n
        elif num_swaps == 3:
            for p1, p2, p3 in itertools.combinations(larger_primes, 3):
                if fixed_prod * p1 * p2 * p3 <= 10**18:
                    primes_list = fixed_primes + [p1, p2, p3]
                    n = fixed_prod * p1 * p2 * p3
                    score = _tau_perp_from_exponents(primes_list, [1]*len(primes_list))
                    if score > best_score:
                        best_score = score
                        best_n = n
    
    # Phase 3: Try 14 primes with one prime squared
    # n = p1^2 * p2 * p3 * ... * p14
    # Product must be <= 10^18
    if time.time() > deadline:
        return best_n
    
    # Try squaring the smallest primes (2, 3, 5) since they have the least impact on product
    for square_prime_idx in range(4):  # Try squaring primes at index 0,1,2,3
        if time.time() > deadline:
            break
        sq_prime = base_primes[square_prime_idx]
        # Use 14 primes total: the squared prime + 13 others
        # The product is sq_prime^2 * (product of 13 other primes)
        # We want to maximize the number of divisors and coprime adjacencies
        
        # Try using the first 14 primes with one squared
        for other_start in range(0, 5):
            if time.time() > deadline:
                break
            # Choose 13 primes from base_primes, excluding the squared one if it's in range
            # Simple approach: use consecutive primes
            candidate_primes = []
            prod = sq_prime ** 2
            # Add 13 more distinct primes
            for i in range(15):
                if i == square_prime_idx:
                    continue
                p = base_primes[i]
                if prod * p > 10**18:
                    break
                candidate_primes.append(p)
                prod *= p
                if len(candidate_primes) == 13:
                    break
            
            if len(candidate_primes) == 13:
                primes_list = [sq_prime] + candidate_primes
                exponents = [2] + [1]*13
                n = prod
                if n <= 10**18:
                    score = _tau_perp_from_exponents(primes_list, exponents)
                    if score > best_score:
                        best_score = score
                        best_n = n
    
    # Phase 4: Try 13 primes with two primes squared
    if time.time() > deadline:
        return best_n
    
    for sq1_idx, sq2_idx in itertools.combinations(range(5), 2):
        if time.time() > deadline:
            break
        sq1 = base_primes[sq1_idx]
        sq2 = base_primes[sq2_idx]
        prod = sq1 ** 2 * sq2 ** 2
        
        # Add 11 more distinct primes
        candidate_primes = []
        for i in range(15):
            if i == sq1_idx or i == sq2_idx:
                continue
            p = base_primes[i]
            if prod * p > 10**18:
                break
            candidate_primes.append(p)
            prod *= p
            if len(candidate_primes) == 11:
                break
        
        if len(candidate_primes) == 11:
            primes_list = [sq1, sq2] + candidate_primes
            exponents = [2, 2] + [1]*11
            n = prod
            if n <= 10**18:
                score = _tau_perp_from_exponents(primes_list, exponents)
                if score > best_score:
                    best_score = score
                    best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
