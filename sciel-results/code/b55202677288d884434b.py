import itertools
import math
import time
import random

def construct():
    t_start = time.time()
    TIME_LIMIT = 80.0  # Safety margin before 90s kill

    # Known valid solution with 9 distinct prime sums (Score 1.8 baseline)
    # A={1, 3, 11}, B={2, 8, 20} ?
    # Let's verify a known good one from previous runs or find a robust one.
    # If search fails, we return a hardcoded 9-sum solution.
    # Example: A=[1, 5, 13], B=[2, 6, 18] -> Sums: 3,7,19, 7,11,23, 15,21,31. 
    # Distinct: 3,7,19,11,23,15,21,31. (8 distinct? 7 repeated).
    # We need 9 distinct.
    # The previous program found one. We will try to find one again, but with relaxed constraints (coprime, not just prime).
    # If we find a 9-sum coprime solution, it's valid.
    
    # Hardcoded fallback that is guaranteed to have 9 distinct coprime sums?
    # Let's use the one the previous program likely found.
    # If we can't guarantee it, we use the search.
    # To ensure we don't drop to 1.0, we will first try to find a 9-sum solution.
    # If found, return it.
    # If not found within a short time, return a known 5-sum solution (Score 1.0) OR try to find a 9-sum one with a different method.
    
    # Actually, the previous program ALWAYS found a 9-sum one in the small range.
    # So we will replicate that search but allow composite coprime sums.
    
    max_val = 256

    # Precompute GCD for all pairs in [2, 256] to speed up coprime checks
    # gcd_matrix[x][y] = math.gcd(x, y)
    # This is 257x257 = 66049 entries. Very fast to compute.
    gcd_matrix = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(1, max_val + 1):
            gcd_matrix[i][j] = math.gcd(i, j)

    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    def check_pair_coprime(A, B):
        """Check if all 9 sums are distinct and pairwise coprime."""
        sums = []
        for a in A:
            for b in B:
                s = a + b
                if s in sums:
                    return False
                sums.append(s)
        
        # Check pairwise coprime
        # sums has 9 elements
        for i in range(9):
            for j in range(i + 1, 9):
                if gcd_matrix[sums[i]][sums[j]] != 1:
                    return False
        return True

    # Strategy 1: Systematic search over small values, allowing composite coprime sums
    search_range = 30
    small_odds = [x for x in odds if x <= search_range]
    small_evens = [x for x in evens if x <= search_range]

    # We prefer prime sums if possible, but coprime is sufficient.
    # The check_pair_coprime is slower than check_pair_prime.
    # To speed up, we can first check if sums are distinct.
    
    for A in itertools.combinations(small_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(small_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair_coprime(A, B):
                return [list(A), list(B)]

    # Try reversed parity
    for A in itertools.combinations(small_evens, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(small_odds, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair_coprime(A, B):
                return [list(A), list(B)]

    # Strategy 2: Expand search range
    search_range = 50
    mid_odds = [x for x in odds if x <= search_range]
    mid_evens = [x for x in evens if x <= search_range]

    for A in itertools.combinations(mid_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(mid_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair_coprime(A, B):
                return [list(A), list(B)]

    for A in itertools.combinations(mid_evens, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(mid_odds, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair_coprime(A, B):
                return [list(A), list(B)]

    # Strategy 3: Random sampling with coprime check
    random.seed(42)
    while time.time() - t_start < TIME_LIMIT:
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        if check_pair_coprime(A, B):
            return [sorted(A), sorted(B)]
        A = random.sample(evens, 3)
        B = random.sample(odds, 3)
        if check_pair_coprime(A, B):
            return [sorted(A), sorted(B)]

    # Fallback: Return a known valid 5-sum solution (Score 1.0)
    # This is worse than 1.8, but better than crashing.
    # However, since the previous program found 1.8 easily, this fallback is unlikely to be hit.
    return [[1, 7, 13], [4, 10, 16]]
