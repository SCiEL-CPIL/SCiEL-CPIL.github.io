# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product
from collections import defaultdict
import heapq

def _tau_perp_from_factors(factors):
    """Compute tau_perp directly from prime factorization dict {p: e}."""
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        pw = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    best_n = 2
    best_score = 0

    # Helper to update best
    def try_candidate(n):
        nonlocal best_n, best_score
        if n < 2 or n > 10**18:
            return
        # Factorize n for scoring
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        s = _tau_perp_from_factors(factors)
        if s > best_score:
            best_score = s
            best_n = n

    # 1. Try primorials and small variations
    # The optimal n likely has many distinct prime factors.
    # Let's search over k = number of distinct primes, from 10 to 15.
    
    for k in range(10, 16):
        if time.time() > deadline - 10:
            break
            
        pk = primes_all[:k]
        base_val = 1
        for p in pk:
            base_val *= p
            
        if base_val > 10**18:
            continue
            
        # Try variations of exponents for the first few primes
        # We assume that adding powers to small primes might increase the score
        # by creating more divisors that are coprime to their neighbors.
        
        # Search space: exponents of 2, 3, 5, 7
        # We limit the search to keep it fast.
        # Max exponent for 2: base_val * 2^a <= 10^18
        max_a = 1
        while base_val * (2 ** max_a) <= 10**18:
            max_a += 1
        max_a = min(max_a, 10) # Cap to prevent explosion
        
        max_b = 1
        while base_val * (3 ** max_b) <= 10**18:
            max_b += 1
        max_b = min(max_b, 8)
        
        max_c = 1
        while base_val * (5 ** max_c) <= 10**18:
            max_c += 1
        max_c = min(max_c, 6)
        
        max_d = 1
        while base_val * (7 ** max_d) <= 10**18:
            max_d += 1
        max_d = min(max_d, 5)

        # We will iterate through a subset of these possibilities.
        # To be efficient, we'll use a greedy approach or limited grid.
        # Given the time limit, we'll just iterate a reasonable range.
        
        for a in range(0, max_a + 1):
            if time.time() > deadline - 5:
                break
            val_a = base_val * (2 ** a)
            if val_a > 10**18:
                break
            for b in range(0, max_b + 1):
                val_ab = val_a * (3 ** b)
                if val_ab > 10**18:
                    break
                for c in range(0, max_c + 1):
                    val_abc = val_ab * (5 ** c)
                    if val_abc > 10**18:
                        break
                    for d in range(0, max_d + 1):
                        val_abcd = val_abc * (7 ** d)
                        if val_abcd > 10**18:
                            break
                        
                        # Construct the number
                        # n = 2^(1+a) * 3^(1+b) * 5^(1+c) * 7^(1+d) * prod(pk[4:])
                        # Note: base_val already has 2^1 * 3^1 * 5^1 * 7^1
                        # So n = base_val * 2^a * 3^b * 5^c * 7^d
                        n_try = val_abcd
                        try_candidate(n_try)
                        
                        # Local search: try swapping an exponent from a small prime to a larger one?
                        # Or just check neighbors in exponent space?
                        # The grid search above covers the main variations.
                        
    # 2. Local refinement around the best found
    if time.time() < deadline - 3:
        # Factor best_n
        best_factors = {}
        temp = best_n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                best_factors[d] = best_factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            best_factors[temp] = best_factors.get(temp, 0) + 1
        
        best_primes = list(best_factors.keys())
        best_exps = [best_factors[p] for p in best_primes]
        
        # Try incrementing/decrementing each exponent by 1
        for i in range(len(best_primes)):
            for delta in [-1, 1]:
                if best_exps[i] + delta >= 1:
                    new_exps = best_exps[:]
                    new_exps[i] += delta
                    n_try = 1
                    for p, e in zip(best_primes, new_exps):
                        n_try *= p ** e
                    if 2 <= n_try <= 10**18:
                        try_candidate(n_try)
                        
        # Try adding a new prime factor if space allows
        # Find smallest prime not in best_factors
        # This is tricky to do efficiently without scanning, but we can check small primes
        for p in primes_all:
            if p not in best_factors:
                n_try = best_n * p
                if n_try <= 10**18:
                    try_candidate(n_try)
                else:
                    break # Primes are increasing, so larger ones will also fail

    return best_n
# EVOLVE-BLOCK-END
