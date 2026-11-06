"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # Product of first 9 primes: 2*3*5*7*11*13*17*19*23 = 223092870
    base = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23
    LIMIT = 10**18
    
    # We want to maximize tau_perp. With all exponents 1,
    # the number of divisors is 2^9 = 512.
    # Adjacent coprime pairs occur between prime powers and their neighbors.
    # 
    # Strategy: Use all 9 small primes with exponent 1.
    # The score is determined by the structure of divisors.
    # 
    # Let's compute the score for the base case first, then try to improve.
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        divs.sort()
        return divs
    
    def tau_perp(n):
        """Count adjacent divisor pairs with gcd 1."""
        if n < 2:
            return 0
        # Factorize
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                factors.append((p, e))
            p += 1 if p == 2 else 2
        if temp > 1:
            factors.append((temp, 1))
        
        divs = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    # Base score with 9 primes, all exponent 1
    base_score = tau_perp(base)
    best_n = base
    best_score = base_score
    
    # Try different combinations: maybe fewer primes but with a large prime factor
    # Or try adding a 10th prime if it fits
    # 2*3*5*7*11*13*17*19*23*29 = 6469693230, still < 10^18
    # 2*3*5*7*11*13*17*19*23*29*31 = 200560470130, still < 10^18
    # 2*3*5*7*11*13*17*19*23*29*31*37 = 7420737394810, still < 10^18
    # 2*3*5*7*11*13*17*19*23*29*31*37*41 = 304250233187210, still < 10^18
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43 = 13082760027050030, still < 10^18
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889721271351410, still < 10^18
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53 = 32589155227381624730 > 10^18
    
    # So we can use up to 15 primes!
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try all subsets of these primes (with exponent 1) to maximize score
    # But 2^15 = 32768 subsets, each scoring takes time
    # Let's be smarter: more primes generally means more divisors and more potential coprime pairs
    
    # Try the full set of 15 primes
    n_15 = 1
    for p in primes_list:
        n_15 *= p
    
    if n_15 <= LIMIT:
        score_15 = tau_perp(n_15)
        if score_15 > best_score:
            best_score = score_15
            best_n = n_15
    
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        return best_n
    
    # Try 14 primes (drop the largest)
    for drop_idx in range(len(primes_list)):
        if time.time() - start_time > DEADLINE - 5:
            break
        n_test = 1
        for i, p in enumerate(primes_list):
            if i != drop_idx:
                n_test *= p
        if n_test <= LIMIT and n_test >= 2:
            s = tau_perp(n_test)
            if s > best_score:
                best_score = s
                best_n = n_test
    
    # Try some combinations with mixed exponents
    # But likely all-1 exponents is optimal for maximizing coprime adjacent pairs
    
    return best_n
# EVOLVE-BLOCK-END
