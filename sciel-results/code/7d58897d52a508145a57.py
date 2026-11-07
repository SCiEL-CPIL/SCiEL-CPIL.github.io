"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

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

def _fast_check_add(S, x):
    """Check if adding x to S creates any violation. S is a set."""
    for a in S:
        g = gcd(a, x)
        rem = a // g
        if rem == 1:
            if len(S) >= 2:
                return False
        else:
            for c in S:
                if c != a and c % rem == 0:
                    return False
    for a in S:
        for b in S:
            if a == b:
                continue
            if (a * b) % x == 0:
                return False
    return True

def _fast_check_add_two(S, x, y):
    """Check if adding both x and y to S creates any violation. S is a set, x,y not in S, x!=y."""
    S2 = S | {x, y}
    # Check x: for each a in S2, a != x, check if a divides x*c for some c in S2, c != a, c != x
    for a in S2:
        if a == x:
            continue
        g = gcd(a, x)
        rem = a // g
        if rem == 1:
            # a divides x, so for any c != a, a divides x*c
            # Need c in S2, c != a, c != x
            # S2 has at least {a, x, y} plus S elements
            # If len(S2) >= 3, there's some c != a and c != x
            if len(S2) >= 3:
                return False
        else:
            for c in S2:
                if c != a and c != x and c % rem == 0:
                    return False
    # Check y: for each a in S2, a != y, check if a divides y*c for some c in S2, c != a, c != y
    for a in S2:
        if a == y:
            continue
        g = gcd(a, y)
        rem = a // g
        if rem == 1:
            if len(S2) >= 3:
                return False
        else:
            for c in S2:
                if c != a and c != y and c % rem == 0:
                    return False
    # Check x divides a*b: for a,b in S2, a != b, a != x, b != x
    for a in S2:
        if a == x:
            continue
        for b in S2:
            if b == x or b == a:
                continue
            if (a * b) % x == 0:
                return False
    # Check y divides a*b: for a,b in S2, a != b, a != y, b != y
    for a in S2:
        if a == y:
            continue
        for b in S2:
            if b == y or b == a:
                continue
            if (a * b) % y == 0:
                return False
    return True

def construct():
    start_time = time.time()
    
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
        primes + composites,
        composites + primes,
        sorted(range(2, 257), key=lambda x: -x),
        sorted(range(2, 257), key=lambda x: x),
        sorted(range(2, 257), key=lambda x: x.bit_count()),
        sorted(range(2, 257), key=lambda x: -x.bit_count()),
    ]
    
    for ordering in orderings:
        S = try_greedy(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        if time.time() - start_time > DEADLINE - 15:
            break
    
    # Iterative refinement: repeatedly try to add elements
    S = set(best_set)
    improved = True
    while improved and time.time() - start_time < DEADLINE - 10:
        improved = False
        for x in composites:
            if x not in S and _fast_check_add(S, x):
                S.add(x)
                improved = True
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # 1-for-1 swap: remove one, add one
    if time.time() - start_time < DEADLINE - 8:
        S = set(best_set)
        for a in list(S):
            for x in composites:
                if x in S:
                    continue
                if _fast_check_add(S - {a}, x):
                    S2 = S - {a}
                    S2.add(x)
                    if len(S2) > len(S):
                        S = S2
                        best_set = S
                        best_size = len(S)
            if time.time() - start_time > DEADLINE - 8:
                break
        # After 1-for-1, try adding more
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for x in composites:
                if x not in S and _fast_check_add(S, x):
                    S.add(x)
                    improved = True
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # 2-for-1 swap: remove one, add two
    if time.time() - start_time < DEADLINE - 5:
        S = set(best_set)
        for a in list(S):
            for x in composites:
                if x in S:
                    continue
                for y in composites:
                    if y in S or y <= x:
                        continue
                    if _fast_check_add_two(S - {a}, x, y):
                        S2 = S - {a}
                        S2.add(x)
                        S2.add(y)
                        if len(S2) > len(S):
                            S = S2
                            best_set = S
                            best_size = len(S)
            if time.time() - start_time > DEADLINE - 5:
                break
        # After 2-for-1, try adding more
        improved = True
        while improved and time.time() - start_time < DEADLINE - 3:
            improved = False
            for x in composites:
                if x not in S and _fast_check_add(S, x):
                    S.add(x)
                    improved = True
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Random restarts with perturbation
    if time.time() - start_time < DEADLINE - 10:
        rng = random.Random(42)
        for trial in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Start from a random subset of primes + random composites
            start_primes = [p for p in primes if rng.random() < 0.8]
            start_comps = [c for c in composites if rng.random() < 0.2]
            ordering = start_primes + start_comps + [x for x in range(2, 257) if x not in start_primes and x not in start_comps]
            rng.shuffle(ordering)
            S = try_greedy(ordering)
            # Refine
            improved = True
            while improved and time.time() - start_time < DEADLINE - 10:
                improved = False
                for x in composites:
                    if x not in S and _fast_check_add(S, x):
                        S.add(x)
                        improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Final: ensure we have at least the primes
    if len(best_set) < best_size:
        best_set = set(primes)
    
    # Verify
    if not _is_valid_set(list(best_set)):
        best_set = set(primes)
    
    return list(best_set)
# EVOLVE-BLOCK-END
