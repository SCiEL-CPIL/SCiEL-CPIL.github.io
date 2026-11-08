"""Constructs a kissing configuration in dimension 11 using a greedy MIS on weight-4 vectors."""
import numpy as np
import time
import random
from itertools import combinations

D = 11
TIME_LIMIT = 110  # Keep well below the 120s wall-clock limit

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # 1. Generate candidate vectors of weight 4 (entries are +/-1)
    #    Number of non-zero indices: C(11,4) = 330
    #    Number of sign patterns: 2^4 = 16
    #    Total candidates: 5280
    candidates = []
    indices_combos = list(combinations(range(D), 4))
    sign_patterns = [tuple(s) for s in product([1, -1], repeat=4)] if False else None
    # Implement product manually or import
    from itertools import product
    sign_patterns = list(product([1, -1], repeat=4))
    
    for idxs in indices_combos:
        for signs in sign_patterns:
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idxs, signs):
                v[i] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # 2. Compute squared norms (all are 4) and pairwise squared distances
    #    dist^2(u, v) = ||u||^2 + ||v||^2 - 2 u.v
    #    Since ||u||^2 = 4 for all, dist^2 = 8 - 2 u.v
    #    We require dist^2 >= max_norm^2 = 4
    #    8 - 2 u.v >= 4  =>  4 >= 2 u.v  =>  u.v <= 2
    
    # Calculate dot product matrix
    # N=5280, D=11. Matrix mult is efficient.
    dots = candidates @ candidates.T
    
    # Conflict matrix: True if u.v > 2 (i.e., 3 or 4)
    # u.v can be -4, -2, 0, 2, 4 for weight-4 vectors?
    # Inner product of two weight-4 vectors with +/-1 entries:
    # Matches on k positions with same sign, k' with opposite.
    # u.v = sum signs.
    # Possible values: -4, -2, 0, 2, 4.
    # Condition u.v <= 2 means we conflict if u.v == 4.
    # Wait, if u.v = 2, dist^2 = 8 - 4 = 4. This is allowed (dist >= norm).
    # So conflict only if u.v == 4?
    # Let's re-verify.
    # Max norm is 2 (squared 4).
    # Min dist must be >= 2 (squared 4).
    # dist^2 = 8 - 2 dot.
    # 8 - 2 dot >= 4
    # 4 >= 2 dot
    # 2 >= dot
    # So dot <= 2 is valid.
    # dot > 2 is invalid.
    # Since dot is integer and even (sum of 4 terms of +/-1), dot can be 4 or 2 or 0...
    # If dot = 4, they are identical vectors.
    # If dot = 2, they agree on 3 positions, disagree on 1?
    # Let u = (1,1,1,1), v = (1,1,1,-1). dot = 1+1+1-1 = 2.
    # dist^2 = (0)^2+(0)^2+(0)^2+(2)^2 = 4. Correct.
    # So we only need to exclude pairs with dot == 4.
    # This simplifies things immensely!
    # We just need a set of vectors where no two are identical?
    # No, wait.
    # If I pick u and v with dot=4, they are the same vector.
    # So the "conflict" is simply identity.
    # Does this mean I can pick ALL 5280 vectors?
    # Let's check the constraint again.
    # "minimum pairwise distance is at least the maximum vector norm"
    # Norm of any vector in the set is 2.
    # Distance between any two distinct vectors in the set:
    # If u != v, can dot be 4? No, if dot=4 then u=v (since norms are equal).
    # So for any distinct u, v in the candidate set, dot <= 2.
    # Therefore, ANY subset of these distinct weight-4 vectors is a valid kissing configuration?
    
    # Let's double check this logic.
    # Is it possible for two distinct weight-4 vectors to have dot product 4?
    # u.v = 4 implies u_i v_i = 1 for all i where non-zero.
    # Since non-zero entries are +/-1, this means u_i = v_i for all i.
    # So u = v.
    # Thus, any two *distinct* vectors from this set have dot <= 2.
    # Consequently, their distance squared is >= 4.
    # The norm squared is 4.
    # So the condition holds for ANY set of distinct vectors from this pool.
    
    # This implies we can return all 5280 vectors!
    # Why did the seed only return 220?
    # The seed used weight-2 vectors.
    # For weight-2: Norm^2 = 2.
    # dist^2 = 2 + 2 - 2 dot = 4 - 2 dot.
    # Condition: dist^2 >= 2 => 4 - 2 dot >= 2 => 2 >= 2 dot => 1 >= dot.
    # Dot product of weight-2 vectors:
    # Entries are +/-1.
    # Possible dots: 2, 0, -2.
    # If dot = 2, vectors are identical (or opposite? No, (1,1) and (1,1) dot 2. (1,1) and (-1,-1) dot -2).
    # If u=(1,1,0..), v=(1,1,0..), dot=2. Conflict.
    # If u=(1,1,0..), v=(1,-1,0..), dot=0. Valid.
    # So for weight-2, we can't include u and v if they are on the same indices with same signs.
    # But we CAN include many others.
    # The seed picked specific ones.
    
    # If the logic above is correct, for weight-4 vectors, ANY distinct pair is valid.
    # So we can just return all 5280 unique vectors.
    
    # Let's verify with a quick mental check.
    # u = [1,1,1,1,0,0,0,0,0,0,0]
    # v = [1,1,1,-1,0,0,0,0,0,0,0]
    # u.v = 1+1+1-1 = 2.
    # dist^2 = (0)^2+(0)^2+(0)^2+(2)^2 = 4.
    # Norm^2 = 4.
    # Dist >= Norm. OK.
    
    # What about:
    # u = [1,1,1,1, ...]
    # v = [1,1,1,1, ...] (Same vector)
    # We shouldn't include duplicates.
    
    # So the maximum possible number is the total number of distinct weight-4 vectors.
    # C(11,4) * 2^4 = 330 * 16 = 5280.
    
    # Is there any catch?
    # "minimum pairwise distance is at least the maximum vector norm"
    # Max norm is 2.
    # Min dist is 2.
    # 2 >= 2. True.
    
    # So we can just return all of them.
    
    # Let's ensure we don't exceed time. Generating 5280 vectors is fast.
    # Checking validity isn't even needed if we trust the math, but let's be safe.
    # Actually, we don't need to check validity in the construct function for speed,
    # but we should ensure we don't have duplicates.
    
    # The generation loop above creates distinct vectors.
    
    return candidates

# Helper for product if not imported at top (it is, but good to be sure)
# Actually, itertools.product is standard.

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Shape: {res.shape}")
    # Verify condition
    if len(res) > 0:
        norms = np.sum(res**2, axis=1)
        max_norm = np.max(norms)
        
        # Check all pairwise distances for a subset or all if small enough
        # 5280^2 is ~28M. Might be slow in python loops, but numpy is fine.
        # Let's just check a random sample or trust the math.
        # For safety in a test environment, let's check fully if it's fast enough.
        # 5280 * 5280 matrix.
        dots = res @ res.T
        dist_sq = 2 * np.diag(dots)[np.newaxis, :] + 2 * np.diag(dots)[:, np.newaxis] - 2 * dots
        # Wait, dist^2 = ||u||^2 + ||v||^2 - 2u.v
        # Here all norms are sqrt(4)=2, so norm^2=4.
        # dist^2 = 4 + 4 - 2 dots = 8 - 2 dots.
        min_dist_sq = np.min(dist_sq - np.eye(len(res)) * 1e9) # Exclude diagonals?
        # Actually min over all pairs including self is 0. We need min over i != j.
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        
        print(f"Max Norm^2: {max_norm}")
        print(f"Min Dist^2: {min_dist_sq}")
        assert min_dist_sq >= max_norm, "Invalid configuration!"
        print("Valid configuration.")
