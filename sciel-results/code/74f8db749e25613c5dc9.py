import math
import time
from math import gcd
from itertools import product
import random

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(1000)

def _tau_perp_fast(n, factorization):
    """Compute tau_perp using factorization for speed.
    factorization: list of (prime, exponent)
    """
    # Generate all divisors from factorization
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _search():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    LIMIT = 10**18
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _tau_perp_fast(best_n, [(2,1),(3,1),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1)])
    
    primes = _PRIMES
    
    # Strategy 1: Try squarefree numbers with first k primes (primorials)
    for k in range(2, 20):
        if time.time() > deadline - 5:
            break
        prod = 1
        factorization = []
        valid = True
        for i in range(k):
            if prod * primes[i] > LIMIT:
                valid = False
                break
            prod *= primes[i]
            factorization.append((primes[i], 1))
        if not valid or k < 2:
            continue
        score = _tau_perp_fast(prod, factorization)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Strategy 2: Explore numbers with various exponent patterns
    for k in range(2, 15):
        if time.time() > deadline - 3:
            break
        
        patterns = []
        
        # All ones (squarefree)
        patterns.append([1]*k)
        
        # Various patterns with one prime having higher exponent
        for e1 in [2, 3, 4, 5]:
            for pos in range(min(k, 4)):
                pat = [1]*k
                pat[pos] = e1
                patterns.append(pat)
        
        # Two primes with higher exponents
        for e1 in [2, 3]:
            for e2 in [2, 3]:
                for p1 in range(min(k, 3)):
                    for p2 in range(p1+1, min(k, 4)):
                        pat = [1]*k
                        pat[p1] = e1
                        pat[p2] = e2
                        patterns.append(pat)
        
        # Try each pattern
        seen = set()
        for pattern in patterns:
            if time.time() > deadline - 2:
                break
            key = tuple(pattern)
            if key in seen:
                continue
            seen.add(key)
            
            prod = 1
            factorization = []
            valid = True
            for i in range(k):
                p = primes[i]
                e = pattern[i]
                pe = p ** e
                if prod * pe > LIMIT:
                    valid = False
                    break
                prod *= pe
                factorization.append((p, e))
            if not valid:
                continue
            
            num_divs = 1
            for _, e in factorization:
                num_divs *= (e + 1)
            if num_divs > 200000:
                continue
            
            score = _tau_perp_fast(prod, factorization)
            if score > best_score:
                best_score = score
                best_n = prod
                
    # Strategy 3: Targeted search for k=13, 14, 15 with higher exponents for small primes
    if time.time() < deadline - 10:
        # k=14
        if time.time() < deadline - 10:
            k = 14
            base_prod = 1
            for i in range(4, k):
                base_prod *= primes[i]
                
            # Iterate exponents for first 4 primes: 2, 3, 5, 7
            max_e0 = 8
            max_e1 = 5
            max_e2 = 4
            max_e3 = 4
            
            for e0 in range(1, max_e0 + 1):
                if time.time() > deadline - 5: break
                p0 = primes[0] ** e0
                if p0 * base_prod > LIMIT: continue
                for e1 in range(1, max_e1 + 1):
                    if time.time() > deadline - 5: break
                    p1 = primes[1] ** e1
                    if p0 * p1 * base_prod > LIMIT: continue
                    for e2 in range(1, max_e2 + 1):
                        if time.time() > deadline - 5: break
                        p2 = primes[2] ** e2
                        if p0 * p1 * p2 * base_prod > LIMIT: continue
                        for e3 in range(1, max_e3 + 1):
                            if time.time() > deadline - 5: break
                            p3 = primes[3] ** e3
                            prod = p0 * p1 * p2 * p3 * base_prod
                            if prod > LIMIT: continue
                            
                            factorization = [
                                (primes[0], e0),
                                (primes[1], e1),
                                (primes[2], e2),
                                (primes[3], e3)
                            ]
                            for i in range(4, k):
                                factorization.append((primes[i], 1))
                                
                            num_divs = 1
                            for _, e in factorization:
                                num_divs *= (e + 1)
                            if num_divs > 500000: continue
                            
                            score = _tau_perp_fast(prod, factorization)
                            if score > best_score:
                                best_score = score
                                best_n = prod
                                
        # k=13
        if time.time() < deadline - 5:
            k = 13
            base_prod = 1
            for i in range(6, k):
                base_prod *= primes[i]
                
            # Iterate exponents for first 6 primes: 2, 3, 5, 7, 11, 13
            max_e0 = 10
            max_e1 = 7
            max_e2 = 5
            max_e3 = 5
            max_e4 = 4
            max_e5 = 4
            
            for e0 in range(1, max_e0 + 1):
                if time.time() > deadline - 3: break
                p0 = primes[0] ** e0
                if p0 * base_prod > LIMIT: continue
                for e1 in range(1, max_e1 + 1):
                    if time.time() > deadline - 3: break
                    p1 = primes[1] ** e1
                    if p0 * p1 * base_prod > LIMIT: continue
                    for e2 in range(1, max_e2 + 1):
                        if time.time() > deadline - 3: break
                        p2 = primes[2] ** e2
                        if p0 * p1 * p2 * base_prod > LIMIT: continue
                        for e3 in range(1, max_e3 + 1):
                            if time.time() > deadline - 3: break
                            p3 = primes[3] ** e3
                            if p0 * p1 * p2 * p3 * base_prod > LIMIT: continue
                            for e4 in range(1, max_e4 + 1):
                                if time.time() > deadline - 3: break
                                p4 = primes[4] ** e4
                                if p0 * p1 * p2 * p3 * p4 * base_prod > LIMIT: continue
                                for e5 in range(1, max_e5 + 1):
                                    if time.time() > deadline - 3: break
                                    p5 = primes[5] ** e5
                                    prod = p0 * p1 * p2 * p3 * p4 * p5 * base_prod
                                    if prod > LIMIT: continue
                                    
                                    factorization = [
                                        (primes[0], e0),
                                        (primes[1], e1),
                                        (primes[2], e2),
                                        (primes[3], e3),
                                        (primes[4], e4),
                                        (primes[5], e5)
                                    ]
                                    for i in range(6, k):
                                        factorization.append((primes[i], 1))
                                        
                                    num_divs = 1
                                    for _, e in factorization:
                                        num_divs *= (e + 1)
                                    if num_divs > 500000: continue
                                    
                                    score = _tau_perp_fast(prod, factorization)
                                    if score > best_score:
                                        best_score = score
                                        best_n = prod

    return best_n

def construct():
    return _search()
# EVOLVE-BLOCK-END
