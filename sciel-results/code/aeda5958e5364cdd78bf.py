"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Search over exponent vectors for products of small primes.
The key insight is that higher powers of small primes combined with
fewer distinct primes can create more divisors with different adjacency
patterns, potentially yielding more coprime adjacent pairs than the
square-free primorial.
"""

# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product
from collections import defaultdict

def _tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    # Generate all divisors
    divs = [1]
    d = 2
    while d * d <= n:
        if n % d == 0:
            for existing in list(divs):
                divs.append(existing * d)
                if d * d != n:
                    divs.append(existing * (n // d))
        d += 1
    if n > 1:
        divs.append(n)
    divs = sorted(set(divs))
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _tau_perp_fast(n):
    """Faster tau_perp using factorization-based divisor generation."""
    if n < 2:
        return 0
    # Factorize n
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
    
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    
    # Generate divisors via recursive product
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
    deadline = start_time + 115  # Safety margin
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Start with primorial of 15 primes
    P15 = 1
    for p in primes_all:
        P15 *= p
    
    best_n = P15
    best_score = _tau_perp_fast(P15)
    
    # Systematic search over exponent vectors
    # For each number of distinct primes k (from 10 to 15), search exponent allocations
    
    for k in range(10, 16):
        if time.time() > deadline - 10:
            break
        
        pk = primes_all[:k]
        
        # Start with all exponents = 1 (square-free)
        base_exps = [1] * k
        base_val = 1
        for p, e in zip(pk, base_exps):
            base_val *= p ** e
        
        if base_val > 10**18:
            continue
        
        # Use a priority queue / best-first search
        # State: (exps tuple, value)
        # We want to explore states that fit in 10^18
        
        visited = set()
        visited.add(tuple(base_exps))
        
        # Use a list as a simple priority queue (sorted by score, then by "potential")
        # Potential: number of primes that can still be incremented
        candidates = [(base_exps[:], base_val)]
        
        # Precompute max exponent for each prime position
        max_exps = []
        for i in range(k):
            max_e = 1
            while base_val * (pk[i] ** max_e) <= 10**18:
                max_e += 1
            max_exps.append(max_e)
        
        # Best-first search: always expand the most promising candidate
        # Promising = high score, or close to being able to increment more primes
        evaluated = set()
        
        max_iterations = 500
        for step in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Pick the best unevaluated candidate
            best_idx = -1
            best_key = None
            for idx, (exps_c, val_c) in enumerate(candidates):
                if tuple(exps_c) in evaluated:
                    continue
                # Score heuristic: prefer higher total exponents (more divisors)
                total_exp = sum(exps_c)
                # Also prefer candidates that can still be improved
                can_improve = any(exps_c[i] < max_exps[i] - 1 for i in range(k))
                key = (can_improve, total_exp)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            exps_c, val_c = candidates[best_idx]
            exps_key = tuple(exps_c)
            evaluated.add(exps_key)
            
            # Evaluate this candidate
            n_c = 1
            for p, e in zip(pk, exps_c):
                n_c *= p ** e
            
            if n_c <= 10**18:
                s = _tau_perp_fast(n_c)
                if s > best_score:
                    best_score = s
                    best_n = n_c
            
            # Generate neighbors: increment each prime's exponent by 1
            for i in range(k):
                if exps_c[i] + 1 < max_exps[i]:
                    new_exps = exps_c[:]
                    new_exps[i] += 1
                    new_val = val_c * pk[i]
                    if new_val <= 10**18:
                        new_key = tuple(new_exps)
                        if new_key not in visited:
                            visited.add(new_key)
                            candidates.append((new_exps, new_val))
            
            # Also try: decrement one and increment another (swap)
            for i in range(k):
                if exps_c[i] > 1:
                    for j in range(k):
                        if i != j and exps_c[j] + 1 < max_exps[j]:
                            new_exps = exps_c[:]
                            new_exps[i] -= 1
                            new_exps[j] += 1
                            new_val = val_c * (pk[j] // pk[i])
                            if new_val <= 10**18:
                                new_key = tuple(new_exps)
                                if new_key not in visited:
                                    visited.add(new_key)
                                    candidates.append((new_exps, new_val))
    
    # Additional targeted searches for specific patterns
    # Pattern 1: 2^a * 3^b * 5^c * 7^d * product of remaining primes
    if time.time() < deadline - 5:
        for k in range(8, 16):
            pk = primes_all[:k]
            base_val = 1
            for p in pk:
                base_val *= p
            
            if base_val > 10**18:
                continue
            
            # Extra budget for 2, 3, 5, 7
            budget = 10**18 // base_val
            
            for a in range(0, min(12, budget // 2 + 1)):
                rem1 = budget // (2 ** a) if a > 0 else budget
                if rem1 < 1:
                    break
                for b in range(0, min(8, rem1 // 3 + 1)):
                    rem2 = rem1 // (3 ** b) if b > 0 else rem1
                    if rem2 < 1:
                        break
                    for c in range(0, min(6, rem2 // 5 + 1)):
                        rem3 = rem2 // (5 ** c) if c > 0 else rem2
                        if rem3 < 1:
                            break
                        for d in range(0, min(4, rem3 // 7 + 1)):
                            extra = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                            n_try = base_val * extra
                            if n_try <= 10**18:
                                if time.time() > deadline - 3:
                                    break
                                s = _tau_perp_fast(n_try)
                                if s > best_score:
                                    best_score = s
                                    best_n = n_try
    
    return best_n
# EVOLVE-BLOCK-END
