"""Search-based program: generate varied candidate (A,B) pairs and keep the best valid one."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    if best_score < 0:
        best_score = 0

    # Build a pool of candidate numbers biased toward primes / small primes
    # to increase chance of pairwise-coprime sums.
    # Include all primes up to 128 and some composites.
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

    primes = [n for n in range(1, 129) if is_prime(n)]
    # Also include 1 and some small composites that are products of distinct small primes
    composites = [6, 10, 14, 15, 21, 22, 26, 33, 34, 35, 38, 39, 46, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95, 105, 106, 110, 114, 115, 118, 119]
    pool = list(set(primes + composites + [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 19, 23, 25, 27, 29, 31, 37]))
    pool = [x for x in pool if 1 <= x <= 128]
    random.shuffle(pool)

    # Try some structured small sets first
    structured = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 3, 5], [2, 4, 6]),
        ([1, 7, 13], [4, 10, 16]),
        ([2, 3, 5], [7, 11, 13]),
        ([1, 2, 4], [3, 5, 7]),
        ([3, 5, 7], [2, 4, 6]),
        ([1, 4, 9], [2, 3, 5]),
        ([2, 5, 11], [3, 7, 13]),
        ([1, 3, 7], [2, 5, 11]),
        ([4, 9, 16], [2, 3, 5]),
    ]
    for A, B in structured:
        if time.time() > deadline:
            break
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]

    # Optimized random search
    # Use a larger pool of primes and small composites
    # Prioritize sets that yield distinct sums
    
    # Pre-compute gcd matrix for pool to speed up coprime check? 
    # Pool is small, direct gcd is fine.
    
    # Try to find a 9-sumset. 
    # Strategy: Pick A, then pick B such that A+B has 9 distinct elements.
    # Then check coprime.
    
    # First, try to maximize distinct sums.
    # A random search for distinct sums is easier.
    
    # Enhanced pool: All numbers 1..128 that are coprime to many small numbers?
    # Just use the existing pool.
    
    # To speed up, we can check distinctness first.
    def distinct_count(A, B):
        return len({a + b for a in A for b in B})
        
    # Run a search focusing on distinctness, then filter by coprime.
    # This is likely faster because distinctness is a weaker constraint than coprimality.
    
    search_start = time.time()
    max_search_time = 70
    
    while time.time() < search_start + max_search_time:
        # Random A, B
        A = random.sample(pool, 3)
        B = random.sample(pool, 3)
        
        # Quick check: if sums are not all distinct, skip coprime check
        sums = [a + b for a in A for b in B]
        if len(set(sums)) < 9:
            continue
            
        # If 9 distinct, check coprime
        if is_pairwise_coprime(set(sums)):
            best_score = 9
            best_A, best_B = A[:], B[:]
            break
        else:
            # Update best if this score is higher than current best (which might be < 9)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]

    # Local refinement: try swapping one element of best_A or best_B
    if best_score < 9 and time.time() < deadline - 5:
        for _ in range(20000):
            if time.time() > deadline - 3:
                break
            A = best_A[:]
            B = best_B[:]
            # Pick a set and an index to mutate
            if random.random() < 0.5:
                idx = random.randrange(3)
                A[idx] = random.choice(pool)
            else:
                idx = random.randrange(3)
                B[idx] = random.choice(pool)
            # Ensure distinct
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                continue
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

    return [best_A, best_B]
# EVOLVE-BLOCK-END
