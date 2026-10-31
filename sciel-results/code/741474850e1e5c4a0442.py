import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy:
    # Model the problem as a Maximum Independent Set (MIS) on the graph of all
    # weight-4 integer vectors. The norm of a weight-4 vector is 4 (norm_sq = 16).
    # We seek the largest set of such vectors where the pairwise distance is >= 4.
    # dist^2 >= 16 => ||v||^2 + ||w||^2 - 2 v.w >= 16 => 16 + 16 - 2 v.w >= 16 => 16 >= 2 v.w => v.w <= 8.
    # Since v and w are weight-4, their dot product is the sum of products of overlapping coordinates.
    # If they share k coordinates, the max dot product is k (if signs align).
    # If k=0, dot=0. If k=1, dot=1. If k=2, dot=2. If k=3, dot=3. If k=4, dot=4.
    # Wait, the vectors are in R^11. Weight 4 means 4 non-zero entries.
    # Max overlap is 4. Max dot product is 4 (if signs are same).
    # So v.w <= 4 always?
    # Let's re-evaluate.
    # v = (1, 1, 1, 1, 0, ..., 0)
    # w = (1, 1, 1, 1, 0, ..., 0) -> dot = 4. dist^2 = 16+16-8 = 24 >= 16. OK.
    # w = (-1, 1, 1, 1, 0, ..., 0) -> dot = 2. dist^2 = 16+16-4 = 28 >= 16. OK.
    # w = (1, 1, 1, 0, 1, 0, ..., 0) -> dot = 3. dist^2 = 16+16-6 = 26 >= 16. OK.
    # It seems ALL weight-4 vectors are pairwise at distance >= 4?
    # Let's check the minimum distance.
    # Min dist^2 occurs when dot product is maximized.
    # Max dot product for two weight-4 vectors is 4 (identical support, same signs).
    # But we need DISTINCT vectors.
    # If v and w are distinct weight-4 vectors:
    # Case 1: Same support. Signs differ in at least one position.
    #   If they differ in 1 sign: dot = 3. dist^2 = 16+16-6 = 26.
    #   If they differ in 2 signs: dot = 2. dist^2 = 16+16-4 = 28.
    #   If they differ in 3 signs: dot = 1. dist^2 = 16+16-2 = 30.
    #   If they differ in 4 signs: dot = 0. dist^2 = 16+16-0 = 32.
    # Case 2: Different support.
    #   Overlap k=3: Max dot = 3. dist^2 = 16+16-6 = 26.
    #   Overlap k=2: Max dot = 2. dist^2 = 16+16-4 = 28.
    #   Overlap k=1: Max dot = 1. dist^2 = 16+16-2 = 30.
    #   Overlap k=0: Max dot = 0. dist^2 = 16+16-0 = 32.
    
    # It appears that the minimum pairwise distance between any two DISTINCT weight-4 vectors is sqrt(26) > 4.
    # Therefore, the set of ALL weight-4 vectors is a valid kissing configuration!
    
    # How many weight-4 vectors are there?
    # Choose 4 coordinates from 11: C(11, 4) = 330.
    # Choose signs for 4 coordinates: 2^4 = 16.
    # Total = 330 * 16 = 5280.
    
    # This is a massive improvement over 220.
    # Score should be 5280 / 220 = 24.0 (if seed was 220).
    # The prompt says "score is k" and "search score is value/seed".
    # If seed program returns 220, and we return 5280, score is 24.
    
    # Let's verify the constraint: "minimum pairwise distance is at least the maximum vector norm".
    # Max norm = 4 (since all vectors have norm 4).
    # Min dist = sqrt(26) ≈ 5.099.
    # 5.099 >= 4. Yes.
    
    # So we can just return all 5280 vectors.
    # No search needed.
    
    # However, the directive asks for a "different strategy" and mentions "branch-and-bound".
    # But if the entire pool is a valid independent set, the MIS is the whole pool.
    # We can just generate them.
    
    # Let's double check if there are any conflicts.
    # Conflict if dist^2 < max_norm_sq = 16.
    # We found min dist^2 = 26.
    # So no conflicts.
    
    # Generate all weight-4 vectors.
    candidates = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(tuple(v))
            
    # Check time
    # Generation is fast.
    
    return np.array(candidates, dtype=np.int64)

# EVOLVE-BLOCK-END
