"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_DEADLINE = time.time() + 115.0  # safety margin before 120s limit

def _is_prime(n):
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

def _get_primes(limit):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _coverage_length(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    # classes: list of [p, r]
    if not classes:
        return 0
    n = len(classes)
    # Use a bytearray for coverage up to cap
    # But cap can be 10^5, which is fine
    covered = bytearray(cap + 1)
    for p, r in classes:
        # Mark multiples: numbers x such that x % p == r
        # x = r, r+p, r+2p, ...
        # But x must be >= 1
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, cap + 1, p):
            covered[x] = 1
    # Find the first gap
    L = 0
    for x in range(1, cap + 1):
        if covered[x]:
            L = x
        else:
            break
    return L

def _reciprocal_sum(classes):
    """Compute sum of 1/p as exact Fraction."""
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    global _DEADLINE
    start_time = time.time()
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We want to select a subset of primes and assign residues to maximize coverage prefix.
    # Constraint: sum of 1/p <= 1.
    
    # Strategy: 
    # 1. Start with a greedy selection of small primes (high coverage density) but we need sum <= 1.
    #    Actually, small primes have large 1/p. We want to pack as much coverage as possible.
    #    The seed used primes from 13 upward. Sum of reciprocals of primes from 13 to ~1000 is < 1.
    #    Let's compute the sum for all primes up to 1000: it's about 2.2, so we can't take all.
    #    
    #    Better approach: 
    #    - Use local search. 
    #    - Initial solution: take the smallest primes such that sum <= 1? 
    #      No, small primes cover more numbers per unit reciprocal? 
    #      Actually, a prime p covers 1/p fraction of numbers. So reciprocal sum is exactly the density.
    #      To cover a prefix, we need to arrange residues so that the union covers 1..L.
    #    
    #    Let's try: 
    #    - Start with all primes up to some limit, but ensure sum <= 1.
    #    - Actually, the sum of reciprocals of primes is divergent, so we can always find a subset.
    #    
    #    Key insight: To maximize the covered prefix L, we want to cover consecutive integers.
    #    The residues should be chosen so that for each x in 1..L, there exists (p,r) in classes with x % p == r.
    #    
    #    Algorithm:
    #    1. Select a set of primes P with sum(1/p) <= 1.
    #    2. Assign residues to maximize the prefix.
    #    
    #    For step 1, we can try different subsets. 
    #    For step 2, we can use local search on residues.
    #    
    #    Let's start with a good initial set: 
    #    - Take primes in increasing order, but skip some to keep sum <= 1.
    #    - Actually, taking smaller primes gives higher density, but they also have more "waste" 
    #      in the sense that their residues are more constrained.
    #    
    #    Let's try: 
    #    - Initial: take all primes p where 1/p is small enough, i.e., larger primes.
    #    - But larger primes cover fewer numbers in a given range.
    #    
    #    Compromise: 
    #    - Use a mix. 
    #    - Start with the seed's approach: primes from 13 upward, but adjust to fit sum <= 1.
    #    
    #    Let's compute: sum of 1/p for p in [13, 1000] is about 0.92 (as per seed comment).
    #    So the seed's set is valid. 
    #    
    #    Now, optimize residues via local search.
    
    # Initial set: primes from 13 to 1000
    initial_primes = [p for p in primes if p >= 13]
    initial_classes = [[p, 0] for p in initial_primes]
    
    # Verify sum
    s = _reciprocal_sum(initial_classes)
    if s > 1:
        # Remove largest primes until sum <= 1
        # Actually, we should remove the ones with largest 1/p? No, we want to keep density high.
        # Let's just use a greedy: start with empty, add primes in order of increasing 1/p (i.e., decreasing p)
        # until sum would exceed 1.
        initial_classes = []
        s = Fraction(0, 1)
        for p in sorted(initial_primes, key=lambda x: -x):  # largest primes first
            if s + Fraction(1, p) <= 1:
                initial_classes.append([p, 0])
                s += Fraction(1, p)
    
    # Now, local search on residues
    best_classes = [c[:] for c in initial_classes]
    best_L = _coverage_length(best_classes)
    
    # Local search: try changing one residue at a time
    # For each class i, try all residues r in 0..p_i-1
    # Keep the best improvement
    
    n = len(best_classes)
    if n == 0:
        return best_classes
    
    # Precompute for speed: for each class, the prime
    primes_list = [c[0] for c in best_classes]
    
    # We'll do multiple passes
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes and time.time() < _DEADLINE - 5:
        improved = False
        passes += 1
        for i in range(n):
            p = primes_list[i]
            old_r = best_classes[i][1]
            best_r = old_r
            best_L_i = best_L
            # Try all residues
            # To speed up, we can try residues in a smart order, but for now try all
            for r in range(p):
                if r == old_r:
                    continue
                # Temporarily change
                best_classes[i][1] = r
                L_new = _coverage_length(best_classes)
                if L_new > best_L_i:
                    best_L_i = L_new
                    best_r = r
                # Restore
                best_classes[i][1] = old_r
            
            if best_r != old_r:
                best_classes[i][1] = best_r
                best_L = best_L_i
                improved = True
    
    # Final check
    final_L = _coverage_length(best_classes)
    final_s = _reciprocal_sum(best_classes)
    if final_s > 1:
        # Should not happen, but just in case, return seed-like solution
        return [[p, (i + 1) % p] for i, p in enumerate([p for p in primes if p >= 13])]
    
    return best_classes
# EVOLVE-BLOCK-END
