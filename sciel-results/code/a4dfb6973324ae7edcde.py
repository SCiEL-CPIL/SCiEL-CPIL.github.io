"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _violates(S, a, b, c):
    """Check if a divides b*c (a, b, c distinct)."""
    return (b * c) % a == 0

def _is_valid_set(S):
    """Check all triples of distinct elements."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if (b * c) % a == 0:
                    return False
    return True

def _count_violations(S):
    """Count number of violating triples (for optimization)."""
    n = len(S)
    count = 0
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if (b * c) % a == 0:
                    count += 1
    return count

def _fast_check_add(S, x):
    """Check if adding x to S creates any violation. S is a set."""
    # For each a in S: check if a divides x*c for any c in S, c != x (x not in S yet)
    # For x: check if x divides a*b for any a,b in S, a != b
    for a in S:
        # a | x * c  for some c in S?
        # a | x*c means: let g = gcd(a, x), then (a/g) | c
        g = __import__('math').gcd(a, x)
        rem = a // g
        if rem == 1:
            # a/g = 1, so a | x, meaning a divides x. Then for ANY c, a | x*c.
            # Violation if there exists c in S (any c, since a, x, c must be distinct)
            # a and x are distinct (x not in S), so need c in S, c != a.
            if len(S) >= 2:  # there's some c != a
                return False
        else:
            # Need some c in S such that rem | c
            for c in S:
                if c != a and c % rem == 0:
                    return False
    # Check x | a*b for any a, b in S, a != b
    for a in S:
        for b in S:
            if a == b:
                continue
            if (a * b) % x == 0:
                return False
    return True

def construct():
    start_time = time.time()
    
    # Start with all primes in 2..256
    from math import isqrt, gcd
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    composites = [x for x in range(2, 257) if not is_prime(x)]
    
    # Strategy: Try multiple greedy orderings and local search
    best_set = set(primes)
    best_size = len(primes)
    
    def try_greedy(candidate_order):
        """Greedily add elements in given order."""
        S = set()
        for x in candidate_order:
            if _fast_check_add(S, x):
                S.add(x)
        return S
    
    # Try different orderings
    orderings = [
        primes + composites,          # primes first
        composites + primes,          # composites first
        sorted(range(2, 257), key=lambda x: -x),  # descending
        sorted(range(2, 257), key=lambda x: x),   # ascending
    ]
    
    for ordering in orderings:
        S = try_greedy(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Local search: try to add composites to best set
    S = set(best_set)
    for x in composites:
        if x not in S and _fast_check_add(S, x):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Try removing one element and adding two others
    if time.time() - start_time < DEADLINE - 5:
        S = set(best_set)
        for a in list(S):
            for x in composites:
                if x in S:
                    continue
                # Try removing a and adding x
                S2 = S - {a}
                S2.add(x)
                if _fast_check_add(S - {a}, x):
                    if len(S2) > len(S):
                        S = S2
                        best_set = S
                        best_size = len(S)
                if time.time() - start_time > DEADLINE - 5:
                    break
            if time.time() - start_time > DEADLINE - 5:
                break
    
    # Random restarts with perturbation
    if time.time() - start_time < DEADLINE - 10:
        rng = random.Random(42)
        for trial in range(200):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Start from a random subset of primes + random composites
            start_primes = [p for p in primes if rng.random() < 0.8]
            start_comps = [c for c in composites if rng.random() < 0.2]
            ordering = start_primes + start_comps + [x for x in range(2, 257) if x not in start_primes and x not in start_comps]
            rng.shuffle(ordering)
            S = try_greedy(ordering)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Final: ensure we have at least the primes
    if len(best_set) < best_size:
        best_set = set(primes)
    
    # Verify
    if not _is_valid_set(list(best_set)):
        # Fallback to primes
        best_set = set(primes)
    
    return list(best_set)
# EVOLVE-BLOCK-END
