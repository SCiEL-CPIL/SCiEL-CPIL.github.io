import time
import math
from itertools import product
from functools import reduce
from math import gcd

def get_divisors(n):
    """Get all divisors of n by factorization."""
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    return sorted(divisors)

def tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list."""
    if n < 2:
        return 0
    divs = get_divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    # First, compute tau_perp for the square-free primorial P_15
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_n = 1
    best_score = 0
    
    # Candidate 1: P_15 (square-free, 15 primes)
    n15 = 1
    for p in primes:
        n15 *= p
    if n15 <= 10**18:
        s = tau_perp(n15)
        if s > best_score:
            best_score = s
            best_n = n15
    
    # Now try variants: increase exponents of small primes, drop large primes
    # For each k from 13 to 15, try exponent vectors for first k primes
    # such that product <= 10^18
    
    # Precompute: for each number of primes k, find max exponents
    # We'll do a targeted search
    
    # Strategy: for k in [12, 13, 14, 15], try exponent vectors
    # where exponents are small (1, 2, 3, 4, 5) for the first few primes
    
    for k in range(12, 16):
        if time.time() > deadline - 10:
            break
        if k > len(primes):
            break
        selected_primes = primes[:k]
        
        # Determine max exponent for each prime
        # Start with all 1s, then try increasing
        # Use a systematic approach: try all combinations of exponents
        # where the product is <= 10^18
        
        # For efficiency, limit exponent range
        # For prime 2: max exponent such that 2^e * (product of other primes) <= 10^18
        # This is complex, so let's do a greedy approach
        
        # Try: base = product of selected primes (all exponents 1)
        base = 1
        for p in selected_primes:
            base *= p
        
        if base > 10**18:
            continue
        
        # Now try increasing exponents of smaller primes
        # For each prime index, try increasing its exponent
        # and see if we can still fit
        
        # Generate candidates by trying exponent vectors
        # Limit to reasonable search space
        
        # For k=15: only square-free fits (base = P_15 ≈ 6.15e17)
        # Can we increase any exponent? 2^2 * 3*5*...*47 = 2 * P_15 ≈ 1.23e18 > 10^18? 
        # P_15 = 614889782588491410, 2*P_15 = 1229779565176982820 > 10^18. No.
        # So for k=15, only square-free works.
        
        # For k=14: base = P_14 = P_15 / 47 ≈ 1.308e16
        # Can increase exponents significantly
        # 2^a * 3^b * 5^c * ... * 43^1 <= 10^18
        # P_14 = 2*3*5*7*11*13*17*19*23*29*31*37*41*43
        # = 13082761331670030
        # 10^18 / P_14 ≈ 76.4, so we can multiply by up to ~76
        # 2^6 = 64, so 2^6 * P_14 / 2 = 2^5 * 3*5*...*43 might work
        # Let's just do a search
        
        # For each k, try exponent vectors with small exponents
        # Use recursion/DFS to explore
        
        def search_exponents(prime_list, max_val, idx=0, current_exp=None, current_prod=1):
            """DFS over exponent vectors."""
            nonlocal best_n, best_score
            if time.time() > deadline - 5:
                return
            
            if idx == len(prime_list):
                # Evaluate this candidate
                if current_prod >= 2 and current_prod <= max_val:
                    # Compute tau_perp
                    s = tau_perp(current_prod)
                    if s > best_score:
                        best_score = s
                        best_n = current_prod
                return
            
            p = prime_list[idx]
            # Try exponents 1, 2, 3, ... until product exceeds max_val
            # Minimum exponent is 1 (we must include this prime)
            prod = current_prod
            for e in range(1, 30):  # max exponent 29
                prod *= p
                if prod > max_val:
                    break
                search_exponents(prime_list, max_val, idx + 1, None, prod)
        
        # This DFS might be too slow for large k. Let's limit.
        # For k >= 13, the search space is large. Let's be smarter.
        
        if k <= 13:
            # Try a limited search
            # For the first few primes, try exponents 1-5
            # For the rest, exponent 1
            # This is still large, so let's do a more targeted approach
            
            # Instead, let's generate specific candidates:
            # 1. Square-free
            # 2. Increase exponent of 2 by 1, 2, 3, ...
            # 3. Increase exponent of 3 by 1, 2, ...
            # 4. Increase exponent of 2 and 3
            # etc.
            
            # Generate candidates by starting from square-free and modifying
            base_cand = 1
            for p in selected_primes:
                base_cand *= p
            
            if base_cand <= 10**18:
                s = tau_perp(base_cand)
                if s > best_score:
                    best_score = s
                    best_n = base_cand
            
            # Try increasing exponents of first few primes
            # For i in range(min(4, k)): try increasing exponent of prime i
            for i in range(min(4, k)):
                for extra in range(1, 8):
                    cand = base_cand * (selected_primes[i] ** extra)
                    if cand > 10**18:
                        break
                    s = tau_perp(cand)
                    if s > best_score:
                        best_score = s
                        best_n = cand
            
            # Try increasing two primes
            for i in range(min(3, k)):
                for j in range(i+1, min(4, k)):
                    for extra_i in range(1, 5):
                        for extra_j in range(1, 5):
                            cand = base_cand * (selected_primes[i] ** extra_i) * (selected_primes[j] ** extra_j)
                            if cand > 10**18:
                                break
                            s = tau_perp(cand)
                            if s > best_score:
                                best_score = s
                                best_n = cand
    
    return best_n
