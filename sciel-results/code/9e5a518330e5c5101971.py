import math
import time
from itertools import combinations

_DEADLINE = 148.0
_START = time.time()

def _tau_perp_from_factorization(primes):
    """primes: list of distinct primes (squarefree n). Return tau_perp(n)."""
    k = len(primes)
    if k == 0:
        return 0
    if k == 1:
        return 1
    n = 1
    for p in primes:
        n *= p
    # divisors via meet-in-the-middle
    half = k // 2
    A = primes[:half]
    B = primes[half:]
    def gen(sub):
        m = len(sub)
        res = [1]
        for p in sub:
            res = [x for x in res] + [x * p for x in res]
        return res
    da = gen(A)
    db = gen(B)
    divs = [a * b for a in da for b in db]
    divs.sort()
    # gcd of adjacent
    cnt = 0
    prev = divs[0]
    for d in divs[1:]:
        if math.gcd(prev, d) == 1:
            cnt += 1
        prev = d
    return cnt

def _tau_perp_general(n, prime_list):
    """Compute tau_perp for n which may have exponents > 1.
    prime_list: list of primes that divide n (in any order).
    Returns the score."""
    if n < 2:
        return 0
    
    # Factorize n using the provided prime_list
    # Determine exponents
    exponents = []
    for p in prime_list:
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        exponents.append((p, count))
    
    if n != 1:
        # Should not happen if prime_list is complete, but handle gracefully
        # If there's a remaining factor, it must be a prime > last in list
        # We add it as a new prime with exponent 1
        exponents.append((n, 1))
        prime_list.append(n)
        
    k = len(exponents)
    if k == 0:
        return 0
    
    # Generate all divisors
    # For efficiency with potentially large k and exponents, 
    # but since we are bounded by 10^18, the number of divisors is manageable.
    divs = [1]
    for p, e in exponents:
        current_divs = list(divs)
        factor = 1
        for _ in range(e):
            factor *= p
            for d in current_divs:
                divs.append(d * factor)
    
    divs.sort()
    
    cnt = 0
    prev = divs[0]
    for d in divs[1:]:
        if math.gcd(prev, d) == 1:
            cnt += 1
        prev = d
    return cnt

def construct():
    global _START
    deadline = _START + _DEADLINE
    max_val = 10**18
    
    # Base primes: first 15 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # The product of the first 15 primes is ~ 6.1e17, which fits.
    # The product of the first 16 primes (including 53) is ~ 3.2e19, which does not fit.
    # So we can use at most 15 distinct primes if they are the smallest ones.
    # However, we might be able to use exponents > 1 for some primes if we drop others.
    
    best_n = 1
    best_score = -1
    
    # 1. Evaluate the pure primorial (exponents all 1)
    primorial = 1
    for p in base_primes:
        primorial *= p
    if primorial <= max_val:
        s = _tau_perp_from_factorization(base_primes)
        if s > best_score:
            best_score = s
            best_n = primorial
    
    # 2. Search for better configurations with exponents > 1
    # Strategy: DFS over primes, assigning exponents.
    # To keep the search space tractable, we limit the number of primes we consider for exponents > 1.
    # Usually, the smallest primes get higher exponents.
    
    # We will perform a limited DFS.
    # State: (index in base_primes, current_n, current_exponents_list)
    # We want to maximize tau_perp.
    
    # Optimization: Pre-compute scores for "squarefree" parts? 
    # No, the interaction is complex.
    
    # Let's try a heuristic search:
    # Start with the primorial.
    # Try to "upgrade" one prime to exponent 2, and drop another prime to make room.
    # Or just DFS with a tight time limit.
    
    def dfs(idx, current_n, exps):
        nonlocal best_n, best_score
        
        if time.time() > deadline:
            return
            
        # Evaluate current state if valid (current_n > 1)
        if current_n > 1 and current_n <= max_val:
            # Reconstruct prime list with exponents for scoring
            # exps is a list of (prime, exp)
            if exps:
                # We need to pass the list of primes and exponents to _tau_perp_general
                # But _tau_perp_general takes n and prime_list.
                # It re-factors. This is expensive if done at every node.
                # Let's optimize: only evaluate at leaves or promising nodes.
                # For now, let's evaluate if we are at the end of the prime list 
                # or if we've decided not to use more primes.
                pass
        
        if idx >= len(base_primes):
            # Leaf node
            if current_n > 1:
                # Compute score
                # We need the list of primes used
                primes_used = [p for p, e in exps if e > 0]
                if primes_used:
                    s = _tau_perp_general(current_n, primes_used)
                    if s > best_score:
                        best_score = s
                        best_n = current_n
            return
        
        p = base_primes[idx]
        
        # Option 1: Skip this prime (exponent 0)
        dfs(idx + 1, current_n, exps)
        
        # Option 2: Use this prime with exponent >= 1
        # Max exponent allowed by max_val
        limit = max_val // current_n
        e_max = 0
        temp = 1
        while temp * p <= limit:
            temp *= p
            e_max += 1
            
        # Try exponents 1 to e_max
        # To keep search space small, we might limit e_max or use a heuristic.
        # However, with only 15 primes, and strict bounds, the branching factor is small.
        # But 2^15 = 32768 leaves is small.
        # The issue is the cost of _tau_perp_general.
        # Generating divisors for n with ~15 primes and small exponents is fast (2^15 divisors max ~ 32k).
        # Sorting 32k ints is fast.
        # So evaluating every leaf is feasible.
        
        for e in range(1, e_max + 1):
            new_n = current_n * (p ** e)
            if new_n > max_val:
                break
            dfs(idx + 1, new_n, exps + [(p, e)])
            
            if time.time() > deadline:
                return

    # Start DFS
    dfs(0, 1, [])
    
    # Ensure we return a valid integer
    if best_n < 2:
        return 2
    return best_n

# EVOLVE-BLOCK-END
