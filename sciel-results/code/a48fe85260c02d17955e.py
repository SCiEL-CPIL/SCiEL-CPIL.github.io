"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 20s safety margin

    # Type D vectors: (1,1,0,...,0), (1,-1,0,...,0) on each pair of coordinates
    # For each pair (i,j), we have 4 vectors: (1,1), (1,-1), (-1,1), (-1,-1)
    # but these are equivalent up to sign in some sense. Actually, let's just enumerate all.
    
    # The current seed generates all 2*2=4 vectors per pair, for C(11,2)=55 pairs -> 220 vectors.
    # But many of these violate the distance constraint.
    # Two vectors v, w are compatible if ||v-w|| >= max(||v||, ||w||).
    # For type D vectors with ||v|| = sqrt(2), we need ||v-w|| >= sqrt(2), i.e., ||v-w||^2 >= 2.
    
    # v and w are type D vectors. 
    # If they share no nonzero coordinates: ||v-w||^2 = 2+2 = 4 >= 2. OK.
    # If they share 1 nonzero coordinate: 
    #   v = e_i + e_j, w = e_i + e_k (j != k). v-w = e_j - e_k, ||v-w||^2 = 2 >= 2. OK.
    #   v = e_i + e_j, w = e_i - e_k. v-w = e_j + e_k, ||v-w||^2 = 2 >= 2. OK.
    #   v = e_i + e_j, w = -e_i + e_k. v-w = 2e_i + e_j - e_k, ||v-w||^2 = 4+1+1 = 6 >= 2. OK.
    #   v = e_i + e_j, w = -e_i - e_k. v-w = 2e_i + e_j + e_k, ||v-w||^2 = 4+1+1 = 6 >= 2. OK.
    #   v = e_i + e_j, w = -e_i + e_j. v-w = 2e_i, ||v-w||^2 = 4 >= 2. OK.
    #   v = e_i + e_j, w = -e_i - e_j = -v. ||v-w||^2 = 4||v||^2 = 8 >= 2. OK.
    #   v = e_i + e_j, w = e_j + e_i = v. Same vector, not distinct.
    #   v = e_i + e_j, w = e_j - e_i. v-w = 2e_i, ||v-w||^2 = 4 >= 2. OK.
    #   v = e_i + e_j, w = -e_j + e_i. v-w = 2e_j, ||v-w||^2 = 4 >= 2. OK.
    #   v = e_i + e_j, w = -e_j - e_i = -v. ||v-w||^2 = 8 >= 2. OK.
    
    # So it seems ALL type D vectors are mutually compatible!
    # Let's verify: the minimum distance between distinct type D vectors is sqrt(2), and the norm is sqrt(2).
    # So the condition ||v-w|| >= ||v|| = sqrt(2) becomes ||v-w||^2 >= 2.
    
    # The minimum ||v-w||^2 for distinct type D vectors:
    # v = e_i + e_j, w = e_i - e_j (same support, different sign pattern). v-w = 2e_j, ||v-w||^2 = 4.
    # v = e_i + e_j, w = e_j + e_i = v. Same.
    # v = e_i + e_j, w = -e_i + e_j. v-w = 2e_i, ||v-w||^2 = 4.
    # v = e_i + e_j, w = e_i + e_k (share i). v-w = e_j - e_k, ||v-w||^2 = 2.
    # So min distance squared is 2, which equals the norm squared. So they are all compatible!
    
    # So we can include ALL type D vectors: C(11,2) * 4 = 55 * 4 = 220.
    # But wait, we also have type A vectors: (2,0,...,0) and (-2,0,...,0), norm = 2.
    # And type B vectors: (1,1,1,0,...,0) type, norm = sqrt(3).
    
    # Let's think about what other vectors we can add.
    
    # Actually, let's reconsider. The seed generates 220 vectors and they all satisfy the constraint.
    # Can we add more?
    
    # Type A: e_i * 2, i.e., (2,0,...,0) and (-2,0,...,0). Norm = 2.
    # Distance from (2,0,...,0) to (1,1,0,...,0): sqrt((2-1)^2 + 1) = sqrt(2) < 2. FAIL.
    # So type A vectors are not compatible with type D vectors.
    
    # What about vectors of the form (1,1,1,0,...,0)? Norm = sqrt(3).
    # Distance to (1,1,0,...,0): sqrt(0+0+1) = 1 < sqrt(3). FAIL.
    
    # So the type D vectors form a large set. Let's count: C(11,2) = 55 pairs, 4 sign patterns each = 220.
    # But actually, (1,1) and (-1,-1) are negatives, and (1,-1) and (-1,1) are negatives.
    # So we have 220 distinct vectors.
    
    # Can we do better? What about the D_11 lattice?
    # D_n lattice: all integer vectors with even sum of coordinates.
    # Minimum norm vectors in D_11: (1,1,0,...,0) with norm sqrt(2).
    # The kissing number of D_11 is... let's think.
    
    # Actually, let's just generate all type D vectors and verify.
    
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    # Now, can we add more vectors? Let's try to add vectors of the form (1,1,1,0,...,0)
    # with appropriate signs, but they need to be compatible with all type D vectors.
    
    # Actually, let's check if we can add the "triple" vectors.
    # For a triple (i,j,k), vector (1,1,1,0,...,0) has norm sqrt(3).
    # Distance to (1,1,0,...,0) on pair (i,j): ||(1,1,1)-(1,1,0)|| = 1 < sqrt(3). FAIL.
    
    # So we can't add those.
    
    # What about vectors with a single 2? Like (2,0,...,0)? Norm = 2.
    # Distance to (1,1,0,...,0) on (i,j): if i is in the pair, ||(2,0)-(1,1)|| = sqrt(1+1) = sqrt(2) < 2. FAIL.
    
    # So it seems 220 is the count from type D. But the known kissing number in 11D is much larger.
    
    # Wait, I think I'm confusing things. The task says "minimum pairwise distance is at least the maximum vector norm".
    # If all vectors have the same norm r, then we need min pairwise distance >= r.
    # This is a "spherical code" type problem but with the constraint that distance >= norm, not distance >= some fixed value.
    
    # Let me re-examine. If we normalize all to norm 2, then we need min pairwise distance >= 2.
    # The kissing number in 11 dimensions is the maximum number of unit spheres that can touch a central unit sphere.
    # That's equivalent to finding the maximum number of unit vectors with pairwise distance >= 1 (since they all touch the origin, 
    # distance between two unit vectors that both touch the origin is... wait, no).
    
    # Actually, the standard kissing number problem: maximum number of non-overlapping unit spheres that can touch a central unit sphere.
    # This is equivalent to maximum number of unit vectors with pairwise angular distance >= 60 degrees, i.e., pairwise dot product <= 0.5.
    # Which means pairwise distance >= sqrt(2 - 2*0.5) = 1. So min pairwise distance >= 1 for unit vectors.
    
    # But our problem says: normalize to norm 2, then min pairwise distance >= 2 (the max norm).
    # For norm-2 vectors, pairwise distance >= 2 means pairwise dot product <= 0 (since ||v-w||^2 = 4 - 2*v.w >= 4 => v.w <= 0).
    # So we need vectors of norm 2 with pairwise dot product <= 0.
    
    # For integer vectors of norm 2: (1,1,0,...,0) type. Dot product of two such vectors:
    # (1,1) on (i,j) and (1,1) on (k,l): dot = 1 if they share one index, 0 if they share none, etc.
    # (1,1) on (i,j) and (1,-1) on (i,j): dot = 1-1 = 0.
    # (1,1) on (i,j) and (1,1) on (i,k): dot = 1 > 0. FAIL!
    
    # So the seed's 220 vectors do NOT all satisfy the constraint!
    # Let me recheck.
    
    # v = e_i + e_j, w = e_i + e_k (j != k, j != i, k != i, j != k).
    # ||v|| = sqrt(2), ||w|| = sqrt(2).
    # v.w = 1.
    # ||v-w||^2 = 2 + 2 - 2*1 = 2.
    # Max norm = sqrt(2).
    # Need ||v-w|| >= max norm => sqrt(2) >= sqrt(2). OK, equality holds.
    
    # So they ARE compatible. The constraint is min pairwise distance >= max norm, and both are sqrt(2).
    
    # But after normalizing to norm 2, the distance becomes 2*sqrt(2)/sqrt(2) = 2, and norm is 2. So distance >= norm. OK.
    
    # So 220 is valid. Can we add more?
    
    # What about vectors like (1,1,1,0,...,0) with norm sqrt(3)?
    # To be compatible, we need ||v - u|| >= sqrt(3) for all type D vectors u.
    # v = e_i + e_j + e_k. u = e_i + e_l (l != j, l != k, l != i).
    # v - u = e_j + e_k - e_l. ||v-u||^2 = 3. sqrt(3) >= sqrt(3). OK.
    # v = e_i + e_j + e_k. u = e_i + e_j. v - u = e_k. ||v-u||^2 = 1 < 3. FAIL.
    
    # So we can't add triple vectors if all pair vectors are present.
    
    # What if we remove some pair vectors to add triple vectors?
    
    # This is getting complex. Let's just go with the 220 type D vectors for now.
    
    arr = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint
    # norms = np.sum(arr * arr, axis=1).astype(np.float64)
    # max_norm = norms.max()
    # for i in range(len(arr)):
    #     for j in range(i+1, len(arr)):
    #         d = np.sum((arr[i] - arr[j])**2)
    #         if d < max_norm:
    #             return arr  # Should not happen
    
    return arr

# EVOLVE-BLOCK-END
