import itertools
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # Max possible distinct sums is 9.
    # We aim for 9.
    
    def is_valid(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if len(distinct_sums) < 9:
            return 0
        
        # Check pairwise coprimality
        for i in range(9):
            for j in range(i + 1, 9):
                if math.gcd(distinct_sums[i], distinct_sums[j]) > 1:
                    return 0
        return 9

    t_start = time.time()
    TIME_LIMIT = 85
    
    # Candidate pools
    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    # Stage 1: Small pool (15 elements) -> C(15,3)^2 ~ 200k checks
    small_odds = odds[:15]
    small_evens = evens[:15]
    
    for A in itertools.combinations(small_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        for B in itertools.combinations(small_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            if is_valid(A, B) == 9:
                return [A, B]
                
    # Stage 2: Medium pool (30 elements) -> C(30,3)^2 ~ 16M checks (might be slow, so check time)
    med_odds = odds[:30]
    med_evens = evens[:30]
    
    for A in itertools.combinations(med_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        for B in itertools.combinations(med_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            if is_valid(A, B) == 9:
                return [A, B]
                
    # Stage 3: Random search
    rng = random.Random(42)
    attempts = 0
    while time.time() - t_start < TIME_LIMIT and attempts < 10000:
        attempts += 1
        A = rng.sample(odds, 3)
        B = rng.sample(evens, 3)
        if is_valid(A, B) == 9:
            return [A, B]
            
    # Fallback to seed (may not be 9 sums, but valid structure)
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
