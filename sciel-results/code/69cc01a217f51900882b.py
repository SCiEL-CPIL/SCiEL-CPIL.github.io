"""Rewritten program: maximize cardinality of S ⊆ [2..256] such that for all distinct a,b,c in S, a ∤ b*c."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt

DEADLINE = 85.0  # seconds wall-clock safety margin

def _valid_set(S):
    """Check if set S satisfies: for all distinct a,b,c in S, a does not divide b*c."""
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def _check_pair_conflict(a, S_without_a):
    """Check if adding a to S_without_a creates a conflict.
    Conflict exists if there are distinct b,c in S_without_a such that a | b*c."""
    lst = list(S_without_a)
    n = len(lst)
    for i in range(n):
        b = lst[i]
        for k in range(n):
            if k == i:
                continue
            c = lst[k]
            if (b * c) % a == 0:
                return True
    return False

def _greedy_construct():
    """Greedy: add numbers in order of increasing 'constraint tightness'."""
    # Strategy: prefer numbers with fewer divisors/multiples in range
    # Sort by number of multiples in [2..256] ascending (fewer conflicts)
    candidates = list(range(2, 257))
    
    # Precompute for each number, how many other numbers x in [2..256] have a|x or x|a or a|x^2 etc.
    # Simpler: sort by number of divisors in range (fewer divisors = less likely to divide products)
    def conflict_score(n):
        # Count how many pairs (b,c) in [2..256] satisfy n | b*c
        count = 0
        for b in range(2, 257):
            for c in range(2, 257):
                if b != c and (b * c) % n == 0:
                    count += 1
        return count
    
    # This is O(256^2) per number = 256*65536 ≈ 16M, too slow for all 255 numbers
    # Use a smarter heuristic: numbers with large prime factors are "safer"
    # A number a is safe if it has a prime factor p > 128 (since b,c <= 256, 
    # for a|b*c, we need p^e | b*c for each prime power p^e || a)
    # If a has a prime factor p > 128, then for a|b*c, both b and c must have p as factor
    # (since p > 128 means p^2 > 256, so p appears at most once in b or c)
    # So a|b*c requires p|b AND p|c, meaning b and c are both multiples of p.
    # Multiples of p in [2..256]: floor(256/p) - 1 (excluding p itself? no, including)
    # Actually multiples of p in [2..256]: floor(256/p) if p>=2
    # If p > 128, floor(256/p) = 1 or 2
    # If floor(256/p) <= 1, then only p itself is a multiple, but we need TWO distinct b,c both divisible by p
    # So if a has a prime factor p where floor(256/p) < 2, i.e., p > 128, then NO two distinct b,c in [2..256]
    # are both divisible by p, so a can NEVER divide b*c for distinct b,c. Such numbers are "free".
    
    free_numbers = []
    for n in range(2, 257):
        has_big_prime = False
        temp = n
        for p in range(2, isqrt(n) + 1):
            while temp % p == 0:
                temp //= p
        if temp > 1:
            if temp > 128:
                has_big_prime = True
        # Also check if n itself is prime > 128
        if has_big_prime:
            free_numbers.append(n)
    
    # Free numbers can always be included
    S = set(free_numbers)
    
    # Now try to add non-free numbers greedily
    non_free = [n for n in range(2, 257) if n not in S]
    # Sort by some heuristic: prefer numbers that are "less constraining"
    # A number a constrains the set if there exist b,c in S with a|b*c
    # Prefer numbers where it's hard for them to divide b*c
    # Heuristic: prefer numbers with larger smallest prime factor, or prime numbers
    
    def smallest_prime_factor(n):
        for p in range(2, isqrt(n) + 1):
            if n % p == 0:
                return p
        return n  # prime
    
    non_free.sort(key=lambda x: (-smallest_prime_factor(x), x))
    
    for n in non_free:
        if not _check_pair_conflict(n, S):
            S.add(n)
    
    return S

def _local_search(S, time_limit):
    """Local search: try single additions, removals, and swaps."""
    S = set(S)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Try adding each missing number
        for n in range(2, 257):
            if n in S:
                continue
            if not _check_pair_conflict(n, S):
                S.add(n)
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing one and adding two
        for remove in list(S):
            S2 = S - {remove}
            added = 0
            for n in range(2, 257):
                if n in S2 or n == remove:
                    continue
                if not _check_pair_conflict(n, S2):
                    S2.add(n)
                    added += 1
            if len(S2) > len(S):
                S = S2
                improved = True
                break
        
        if improved:
            continue
        
        # Try random swaps
        for _ in range(200):
            if (time.time() - start) > time_limit:
                break
            if len(S) < 2:
                break
            a = random.choice(list(S))
            b = random.choice([n for n in range(2, 257) if n not in S]) if any(n not in S for n in range(2, 257)) else None
            if b is None:
                break
            S2 = S - {a} | {b}
            if _valid_set(S2):
                S = S2
                improved = True
                break
    
    return S

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    S = _greedy_construct()
    
    # Phase 2: Local search with time budget
    remaining = DEADLINE - (time.time() - start_time) - 5.0  # 5s safety margin
    if remaining > 1.0:
        S = _local_search(S, remaining)
    
    # Convert to list of Python ints
    result = [int(x) for x in sorted(S)]
    
    # Final validation
    if not _valid_set(set(result)):
        # Fallback to primes only (known valid)
        result = [a for a in range(2, 257) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    return result
# EVOLVE-BLOCK-END
