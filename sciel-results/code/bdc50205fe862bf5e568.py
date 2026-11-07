"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
NMAX = 256

def _primes_upto(n):
    """Return list of primes up to n using sieve."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = b"\x00" * len(sieve[p * p::p])
    return [i for i in range(2, n + 1) if sieve[i]]

def _valid(S):
    """Check if set S satisfies the condition: for all distinct a,b,c in S, a does not divide b*c."""
    L = list(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            b = L[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if (b * L[k]) % a == 0:
                    return False
    return True

def _construct_maximal_set():
    """
    Construct a large valid set by:
    1. Start with all primes up to NMAX
    2. Try to add composites that don't violate the condition
    
    Key insight: If we have all primes P, and we want to add a composite c,
    we need to ensure no prime p in P divides c * p2 for any other prime p2 in P.
    Since p, p2 are distinct primes, p | p2 is impossible, so p | c * p2 implies p | c.
    But if p | c, then since c is composite and p is prime, this is possible.
    
    Wait, let me reconsider. The condition is: for all DISTINCT a, b, c in S, a does not divide b*c.
    
    If S contains all primes P and we add composite c:
    - For a = p (prime in P), b = c, k = p2 (another prime in P): need p does not divide c * p2.
      Since gcd(p, p2) = 1, this means p does not divide c.
    - For a = c, b = p1, k = p2: need c does not divide p1 * p2. Since c is composite >= 4 and p1*p2 
      can be small, this might fail.
    
    So the constraint is: for any composite c we add, no prime in P can divide c.
    But that's impossible if c has any prime factor!
    
    Alternative approach: Don't include all primes. Instead, build a set where composites can coexist.
    
    Let's think about the structure differently.
    
    Observation: If we take only numbers that are pairwise "coprime-like" in a strong sense...
    
    Actually, let's try a greedy approach but more carefully.
    
    Start with empty set. Add numbers from 2 to NMAX in some order, adding a number x only if 
    for all pairs (y, z) already in the set (y != z), x does not divide y*z, and for all y, z in set,
    y does not divide x*z, and z does not divide x*y.
    """
    pass  # We'll implement a proper version below

def _greedy_add(S, x, deadline):
    """Check if x can be added to set S without violating the condition."""
    L = list(S)
    n = len(L)
    # Check: for all distinct y, z in L, x does not divide y*z
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (L[i] * L[j]) % x == 0:
                return False
    # Check: for all y in L, y does not divide x * z for any z in L, z != y
    for i in range(n):
        y = L[i]
        for j in range(n):
            if j == i:
                continue
            if (x * L[j]) % y == 0:
                return False
    return True

def _construct_greedy(deadline):
    """Greedy construction: try to add numbers in order of decreasing 'difficulty'."""
    # Sort candidates: prefer numbers with fewer small prime factors first?
    # Or just go in natural order?
    
    # Let's try adding primes first, then composites
    all_primes = _primes_upto(NMAX)
    S = set()
    
    # Add primes first
    for p in all_primes:
        if time.time() > deadline:
            break
        if _greedy_add(S, p, deadline):
            S.add(p)
    
    # Now try to add composites
    composites = [x for x in range(4, NMAX + 1) if x not in set(all_primes)]
    # Sort by number of prime factors (more factors = harder to add?)
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    composites.sort(key=lambda x: num_prime_factors(x))
    
    for c in composites:
        if time.time() > deadline:
            break
        if c in S:
            continue
        if _greedy_add(S, c, deadline):
            S.add(c)
    
    return S

def _local_search_improve(S, deadline):
    """Try to improve the set by replacing elements or adding new ones."""
    S = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each candidate not in S
        for cand in range(2, NMAX + 1):
            if cand in S:
                continue
            if _greedy_add(set(S), cand, deadline):
                S.append(cand)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one element, add another
        for i in range(len(S)):
            for cand in range(2, NMAX + 1):
                if cand in S:
                    continue
                T = S[:i] + S[i + 1:] + [cand]
                if _valid(set(T)):
                    S = T
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    deadline = time.time() + 80.0  # Leave 10s safety margin for 90s limit
    all_primes = _primes_upto(NMAX)
    best_set = set(all_primes)
    best_size = len(all_primes)
    
    # Verify all primes is valid
    if _valid(all_primes):
        pass  # It should be valid
    else:
        # Find a valid subset of primes
        S = set()
        for p in all_primes:
            if _greedy_add(S, p, deadline):
                S.add(p)
        best_set = S
        best_size = len(S)
    
    # Try greedy construction
    if time.time() < deadline - 10:
        S_greedy = _construct_greedy(deadline - 10)
        if len(S_greedy) > best_size:
            if _valid(S_greedy):
                best_set = S_greedy
                best_size = len(S_greedy)
    
    # Try local search improvement
    if time.time() < deadline - 5:
        S_improved = _local_search_improve(list(best_set), deadline - 5)
        if len(S_improved) > best_size:
            if _valid(set(S_improved)):
                best_set = set(S_improved)
                best_size = len(S_improved)
    
    # Final validation
    if not _valid(best_set):
        # Fallback to just primes
        return list(all_primes)
    
    return list(best_set)
# EVOLVE-BLOCK-END
