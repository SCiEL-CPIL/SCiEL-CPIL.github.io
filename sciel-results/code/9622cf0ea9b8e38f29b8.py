"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True

def _valid_triple(a, b, c):
    """Check that a does not divide b*c for distinct a,b,c."""
    return (b * c) % a != 0

def _is_set_valid(S):
    """Check if set S satisfies the triple condition."""
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S while maintaining validity."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            # Check triples: (x, a, b), (a, x, b), (a, b, x)
            if (a * b) % x == 0:
                return False
            if (x * b) % a == 0:
                return False
            if (x * a) % b == 0:
                return False
    return True

def _greedy_construct():
    """Greedy: start with primes, try to add composites."""
    primes = [a for a in range(2, 257) if _is_prime(a)]
    composites = [a for a in range(2, 257) if not _is_prime(a)]
    
    # Sort composites: prefer those with large prime factors (harder to divide)
    def composite_score(x):
        # Score by largest prime factor - larger is better (less likely to cause conflicts)
        lp = 1
        for p in range(2, isqrt(x) + 1):
            while x % p == 0:
                lp = p
                x //= p
        if x > 1:
            lp = max(lp, x)
        return -lp  # negate for sorting (larger lp = higher priority)
    
    composites.sort(key=composite_score)
    
    S = primes[:]
    for x in composites:
        if _can_add(S, x):
            S.append(x)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s safety margin
    
    # Start with greedy
    S = _greedy_construct()
    base_size = len(S)
    
    # Try local search: remove one element, try to add two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_set = set(S)
        for i in range(len(S)):
            if time.time() > deadline:
                break
            # Try removing S[i]
            S_minus = [x for j, x in enumerate(S) if j != i]
            removed = S[i]
            # Try adding composites not in S
            for x in range(2, 257):
                if time.time() > deadline:
                    break
                if x in S_set or x == removed:
                    continue
                if _can_add(S_minus, x):
                    S_new = S_minus + [x]
                    if len(S_new) > len(S):
                        S = S_new
                        S_set = set(S)
                        improved = True
                        break
            if improved:
                break
    
    # Final validation
    if not _is_set_valid(S):
        # Fallback to primes
        S = [a for a in range(2, 257) if _is_prime(a)]
    
    return list(S)
# EVOLVE-BLOCK-END
