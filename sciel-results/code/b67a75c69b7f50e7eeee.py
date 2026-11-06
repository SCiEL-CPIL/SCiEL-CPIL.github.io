"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Uses efficient divisor generation with bitset-based coprime checking
and a broader search over prime exponent configurations.
"""
import math
import time
import itertools


def _tau_perp_from_factorization(primes, exponents):
    """Compute tau_perp(n) given its prime factorization.
    
    Args:
        primes: list of distinct primes
        exponents: list of corresponding exponents
    
    Returns:
        Number of adjacent coprime pairs in the sorted divisor list.
    """
    # For square-free case, use bitset approach
    if all(e == 1 for e in exponents):
        k = len(primes)
        # Each divisor corresponds to a bitmask of which primes divide it
        # Generate all divisors with their masks
        divs = [(1, 0)]
        for i, p in enumerate(primes):
            new_divs = []
            for d, mask in divs:
                new_divs.append((d, mask))
                new_divs.append((d * p, mask | (1 << i)))
            divs = new_divs
        
        divs.sort(key=lambda x: x[0])
        
        count = 0
        for i in range(len(divs) - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                count += 1
        return count
    
    # General case
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        p_powers = [1]
        for _ in range(e):
            p_powers.append(p_powers[-1] * p)
        for d in divisors:
            for pp in p_powers:
                new_divs.append(d * pp)
        divisors = new_divs
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Generate primes up to ~200 for broader search
    def _primes_up_to(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    all_primes = _primes_up_to(300)
    
    best_n = 1
    best_score = 0
    
    def try_candidate(primes_list, exponents_list):
        nonlocal best_n, best_score
        n = 1
        for p, e in zip(primes_list, exponents_list):
            n *= p ** e
        if 2 <= n <= 10**18:
            score = _tau_perp_from_factorization(primes_list, exponents_list)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Phase 1: Square-free with 15 primes - try various prime sets
    # Start with smallest 15 primes
    p15 = all_primes[:15]
    try_candidate(p15, [1] * 15)
    
    # Phase 2: Systematic search over 15-prime square-free sets
    # Strategy: fix first k primes, search for remaining 15-k primes
    for num_fixed in range(10, 14):
        if time.time() > deadline:
            break
        fixed_primes = all_primes[:num_fixed]
        fixed_prod = 1
        for p in fixed_primes:
            fixed_prod *= p
        
        remaining_needed = 15 - num_fixed
        min_prime = all_primes[num_fixed]
        
        # Greedy: try to pick the smallest possible primes for remaining
        # Then try swapping larger ones
        def _search_remaining(start_idx, count_needed, current_primes, current_prod):
            nonlocal best_n, best_score
            if time.time() > deadline:
                return
            if count_needed == 0:
                if len(current_primes) == 15:
                    try_candidate(current_primes, [1] * 15)
                return
            if len(current_primes) + (len(all_primes) - start_idx) < count_needed:
                return
            
            for i in range(start_idx, len(all_primes)):
                p = all_primes[i]
                new_prod = current_prod * p
                if new_prod > 10**18:
                    break
                _search_remaining(i + 1, count_needed - 1, current_primes + [p], new_prod)
        
        _search_remaining(num_fixed, remaining_needed, fixed_primes, fixed_prod)
    
    # Phase 3: 14 primes with one prime squared
    if time.time() > deadline:
        return best_n
    
    for sq_prime in all_primes[:8]:
        if time.time() > deadline:
            break
        sq_prod = sq_prime ** 2
        
        # Find 13 other primes (not including sq_prime) that minimize product
        other_primes = [p for p in all_primes if p != sq_prime]
        
        # Try consecutive sets of 13 from other_primes
        for start in range(0, min(10, len(other_primes) - 13)):
            if time.time() > deadline:
                break
            candidate = other_primes[start:start+13]
            prod = sq_prod
            for p in candidate:
                prod *= p
            if prod <= 10**18:
                primes_list = candidate + [sq_prime]
                primes_list.sort()
                exponents_list = [2 if p == sq_prime else 1 for p in primes_list]
                try_candidate(primes_list, exponents_list)
    
    # Phase 4: 13 primes with two primes squared
    if time.time() > deadline:
        return best_n
    
    for sq1 in all_primes[:6]:
        for sq2 in all_primes[:6]:
            if sq1 >= sq2:
                continue
            if time.time() > deadline:
                break
            sq_prod = sq1 ** 2 * sq2 ** 2
            
            other_primes = [p for p in all_primes if p != sq1 and p != sq2]
            
            for start in range(0, min(8, len(other_primes) - 11)):
                if time.time() > deadline:
                    break
                candidate = other_primes[start:start+11]
                prod = sq_prod
                for p in candidate:
                    prod *= p
                if prod <= 10**18:
                    primes_list = candidate + [sq1, sq2]
                    primes_list.sort()
                    exponents_list = [2 if p in (sq1, sq2) else 1 for p in primes_list]
                    try_candidate(primes_list, exponents_list)
    
    # Phase 5: Try 12 primes with three squared
    if time.time() > deadline:
        return best_n
    
    for sq1 in all_primes[:5]:
        for sq2 in all_primes[:5]:
            for sq3 in all_primes[:5]:
                if not (sq1 < sq2 < sq3):
                    continue
                if time.time() > deadline:
                    break
                sq_prod = sq1 ** 2 * sq2 ** 2 * sq3 ** 2
                
                other_primes = [p for p in all_primes if p not in (sq1, sq2, sq3)]
                
                for start in range(0, min(6, len(other_primes) - 9)):
                    if time.time() > deadline:
                        break
                    candidate = other_primes[start:start+9]
                    prod = sq_prod
                    for p in candidate:
                        prod *= p
                    if prod <= 10**18:
                        primes_list = candidate + [sq1, sq2, sq3]
                        primes_list.sort()
                        exponents_list = [2 if p in (sq1, sq2, sq3) else 1 for p in primes_list]
                        try_candidate(primes_list, exponents_list)
    
    return best_n
