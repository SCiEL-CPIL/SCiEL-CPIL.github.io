"""Rewritten program: search 3x3 additive grids with pairwise-coprime distinct sums."""
import math
import time
import itertools
import random

# Deadline guard: evaluator kills at 90s; use 75s as safety margin.
_DEADLINE = time.time() + 75.0

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(values):
    """Check that all distinct values in the list are pairwise coprime."""
    vals = list(set(values))
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    """Score = number of distinct values in A+B, if pairwise coprime; else 0."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if len(distinct) == 1:
        return 0
    if _pairwise_coprime(list(distinct)):
        return len(distinct)
    return 0

def _try_pair(A, B):
    """Quick validation and scoring for a candidate pair."""
    # A and B must each have exactly 3 distinct integers in [1,128]
    if len(A) != 3 or len(B) != 3:
        return 0
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0
    for x in A:
        if not (1 <= x <= 128):
            return 0
    for x in B:
        if not (1 <= x <= 128):
            return 0
    return _score(A, B)

def _construct():
    """Main search: find A, B (each 3 distinct ints in [1,128]) maximizing
    the number of distinct pairwise-coprime values in A+B."""
    
    # --- Phase 1: Fast exhaustive search over small structured families ---
    
    # Strategy: pick A = {a0, a0+d1, a0+d2}, B = {b0, b0+e1, b0+e2}
    # The 9 sums are: a0+b0 + {0, e1, e2, d1, d1+e1, d1+e2, d2, d2+e1, d2+e2}
    # We want these 9 offsets to yield pairwise-coprime sums.
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Try to find configurations where all 9 sums are distinct and pairwise coprime.
    # A powerful approach: make all 9 sums be primes (primes > 2 are pairwise coprime).
    
    # Precompute: for each pair (A, B) with A,B in [1,128]^3, check if all 9 sums
    # are pairwise coprime. But that's 128^6 ≈ 4.5e12 — too many.
    
    # Instead: iterate over A (128 choose 3 ≈ 348k) and for each A, iterate over B.
    # 348k * 348k ≈ 1.2e11 — still too many.
    
    # Better: iterate over A (348k), and for each A, try to find B such that
    # the 9 sums are pairwise coprime. For a fixed A, the constraints on B are:
    # For each pair of sums (a_i + b_j, a_k + b_l) with (i,j) != (k,l),
    # gcd(a_i + b_j, a_k + b_l) = 1.
    # This is a system of congruence-like constraints on b_1, b_2, b_3.
    
    # Even better: iterate over all 3-element subsets of [1,128] for A,
    # and for each, do a randomized search over B.
    
    # Actually, let's do a smarter structured search.
    
    # Key insight: if we can make all 9 sums prime, they're automatically
    # pairwise coprime. Let's search for A, B where a_i + b_j is prime for all i,j.
    
    # Precompute primes up to 256 (max sum = 128+128=256)
    max_sum = 256
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(max_sum**0.5) + 1):
        if is_prime[p]:
            for m in range(p*p, max_sum + 1, p):
                is_prime[m] = False
    
    # Phase 1a: Search for A, B where all 9 sums are prime.
    # For a fixed A = {a1, a2, a3}, we need b1, b2, b3 such that
    # a_i + b_j is prime for all i, j.
    # This means: for each j, b_j must be such that a1+b_j, a2+b_j, a3+b_j are all prime.
    # And for different j values, the sums must also be pairwise coprime (automatically if all prime).
    
    # Iterate over A (sorted triples from 1..128):
    # For efficiency, limit the search space.
    
    # Number of triples: C(128,3) = 128*127*126/6 = 348,976
    # For each, we need to find b1, b2, b3 in [1,128] such that all 9 sums are prime.
    # For a fixed A, the set of valid b values for each row constraint:
    # valid_b_for_col_j = {b : a1+b, a2+b, a3+b all prime}
    # We need to pick 3 distinct values from the intersection... wait, each b_j is independent.
    # Actually b1, b2, b3 just need to be distinct and in [1,128], and for each j,
    # a_i + b_j is prime for all i.
    
    # So for fixed A, compute S = {b in [1,128] : a1+b, a2+b, a3+b all prime}
    # Then we need |S| >= 3 to pick 3 distinct b values.
    
    # Let's do this:
    
    all_A = []
    for a1 in range(1, 129):
        for a2 in range(a1 + 1, 129):
            for a3 in range(a2 + 1, 129):
                all_A.append((a1, a2, a3))
    
    # That's 348,976 triples. For each, checking all 128 b-values: 348976 * 128 ≈ 44.7M checks.
    # Each check involves 3 prime lookups. Should be fast enough in Python with precomputed is_prime.
    
    # But we also need the sums to be distinct (for maximum cardinality).
    # If all 9 sums are prime, they could still have duplicates.
    # For maximum cardinality, we want all 9 sums distinct.
    
    # Let's first find any A, B where all 9 sums are prime (gives pairwise coprime).
    # Then among those, maximize distinct count.
    
    found_prime_grid = False
    best_prime_score = 0
    best_prime_A = None
    best_prime_B = None
    
    for A in all_A:
        if time.time() > _DEADLINE:
            break
        a1, a2, a3 = A
        # Find all b in [1,128] such that a1+b, a2+b, a3+b are all prime
        valid_bs = []
        for b in range(1, 129):
            if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                valid_bs.append(b)
        
        if len(valid_bs) >= 3:
            # Try all combinations of 3 distinct b values
            for B in itertools.combinations(valid_bs, 3):
                b1, b2, b3 = B
                sums = [a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3]
                distinct_sums = set(sums)
                # All are prime, so pairwise coprime. Score = number of distinct sums.
                sc = len(distinct_sums)
                if sc > best_prime_score:
                    best_prime_score = sc
                    best_prime_A = list(A)
                    best_prime_B = list(B)
                    if sc == 9:
                        found_prime_grid = True
                        # 9 is the maximum possible; we can stop
                        break
            if found_prime_grid:
                break
        if time.time() > _DEADLINE:
            break
    
    if found_prime_grid:
        return [best_prime_A, best_prime_B]
    
    # If we found a prime grid with < 9 distinct, keep it as a candidate
    if best_prime_A is not None:
        best_score = best_prime_score
        best_A = best_prime_A
        best_B = best_prime_B
    
    # Phase 1b: Search for A, B where all 9 sums are pairwise coprime (not necessarily prime).
    # This is harder. Let's do a randomized search.
    
    # Phase 2: Randomized search
    rng = random.Random(42)
    
    # Try random pairs
    for _ in range(2000000):
        if time.time() > _DEADLINE:
            break
        A = rng.sample(range(1, 129), 3)
        B = rng.sample(range(1, 129), 3)
        sc = _try_pair(A, B)
        if sc > best_score:
            best_score = sc
            best_A = A
            best_B = B
            if best_score == 9:
                return [best_A, best_B]
    
    # Phase 3: Local search / hill climbing from best found
    if best_score > 0:
        for _ in range(500000):
            if time.time() > _DEADLINE:
                break
            # Perturb best_A or best_B
            if rng.random() < 0.5:
                A = list(best_A)
                idx = rng.randrange(3)
                A[idx] = rng.randint(1, 128)
                B = list(best_B)
            else:
                A = list(best_A)
                B = list(best_B)
                idx = rng.randrange(3)
                B[idx] = rng.randint(1, 128)
            
            sc = _try_pair(A, B)
            if sc > best_score:
                best_score = sc
                best_A = A
                best_B = B
                if best_score == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime. Maximize the cardinality of A+B."""
    return _construct()
