"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a kissing configuration in R^11 with integer vectors.
    Returns an integer array of shape (k, D).
    
    Strategy: Use a product of known optimal/suboptimal kissing configurations.
    In R^11, we can embed configurations from lower dimensions.
    
    One strong approach: Use the E8 lattice structure.
    The kissing number of R^11 is known to be at least 252 (from 240 E8 + 12 D3? or other constructions).
    
    Let's use a greedy construction with integer vectors, starting from known good sets.
    
    Known optimal kissing numbers:
    k(2) = 6
    k(3) = 12
    k(4) = 24
    k(5) = 40
    k(6) = 72
    k(7) = 126
    k(8) = 240
    k(9) = 460 (or more?)
    k(10) = ?
    k(11) = ?
    
    A simple product construction: If we have a kissing configuration of size k1 in R^m and size k2 in R^n,
    we can form a configuration in R^(m+n) by taking products. But this isn't directly additive.
    
    Alternative: Start with the E8 lattice in R^8, then add vectors in the remaining 3 dimensions.
    E8 has 240 minimal vectors of norm sqrt(2).
    If we embed E8 in R^11 by adding 3 zero coordinates, we get 240 vectors.
    Then we can add vectors that are nonzero only in the last 3 coordinates.
    In R^3, the kissing number is 12. These 12 vectors have norm sqrt(2) (if we use integer vectors like (1,1,0) type).
    
    Let's check distances:
    - Between two E8 vectors (embedded): distance >= sqrt(2) (since minimal norm is sqrt(2), min distance is sqrt(2) for orthogonal? No, min distance between distinct minimal vectors in E8 is sqrt(2) only if they are orthogonal. Actually, the minimum distance between distinct minimal vectors in E8 is sqrt(2) for some pairs, but can be larger. Wait, the minimal vectors have norm sqrt(2). The distance between two distinct minimal vectors v and w is |v-w|. Since v.w can be -1, 0, or 1, |v-w|^2 = 2+2-2v.w. If v.w=1, dist=0 (same vector). If v.w=0, dist=sqrt(4)=2. If v.w=-1, dist=sqrt(8)=2sqrt(2). So min distance between distinct E8 minimal vectors is 2.
    - Between an E8 vector (a, 0,0,0) and a D3 vector (0, b, c, d) where (b,c,d) is a D3 minimal vector: distance^2 = |a|^2 + |b|^2 = 2+2=4, so distance=2.
    - Between two D3 vectors: min distance is 2 (similar to E8).
    
    So the minimum distance is 2, and the maximum norm is sqrt(2). 
    Wait, the condition is: min pairwise distance >= max vector norm.
    Here, min distance = 2, max norm = sqrt(2) ≈ 1.414. So 2 >= 1.414, which holds.
    
    But we can scale. The vectors are integers. 
    E8 minimal vectors: 112 of type (±1, ±1, 0, ..., 0) and 128 of type (±1/2, ..., ±1/2) with even number of -1s.
    To make them integers, multiply the half-integer ones by 2. But then the norms differ.
    
    Let's use a different set. The "even" subset of the hypercube: all vectors with coordinates ±1 and an even number of -1s. There are 2^7 = 128 such vectors in R^8. Their norm is sqrt(8) = 2sqrt(2).
    The "odd" subset: all vectors with coordinates ±1 and an odd number of -1s. Also 128 vectors.
    
    If we take all 256 hypercube vertices in R^8, the minimum distance between distinct vertices is 2 (since they differ in at least 2 coordinates? No, they can differ in 1 coordinate. If they differ in 1 coordinate, distance is 2. If they differ in 2, distance is 2sqrt(2), etc. So min distance is 2. Max norm is sqrt(8) = 2sqrt(2) ≈ 2.828. 
    Condition: min distance >= max norm? 2 >= 2.828? No. So the full hypercube doesn't work directly.
    
    However, if we take only one coset (say, even parity), the minimum distance between distinct vectors in the coset is 2sqrt(2) (since they must differ in an even number of coordinates, minimum 2). Max norm is 2sqrt(2). So 2sqrt(2) >= 2sqrt(2) holds.
    So the 128 even-parity hypercube vertices form a valid configuration in R^8.
    
    Now, add 3 more dimensions. We can add the 12 minimal vectors of D3 (which are (±1,±1,0), (±1,0,±1), (0,±1,±1)) embedded in the last 3 coordinates. These have norm sqrt(2).
    But we need to check distances between the R^8 part and the R^3 part.
    Let v be in R^8 (even parity, norm 2sqrt(2)) and w be in R^3 (norm sqrt(2)).
    Distance^2 = |v|^2 + |w|^2 = 8 + 2 = 10. Distance = sqrt(10) ≈ 3.16.
    Max norm in the combined set: max(2sqrt(2), sqrt(2)) = 2sqrt(2) ≈ 2.828.
    Min distance in R^8 part: 2sqrt(2) ≈ 2.828.
    Min distance in R^3 part: sqrt(2) * sqrt(2) = 2? Wait, min distance between distinct D3 minimal vectors is 2 (e.g., (1,1,0) and (1,-1,0) differ by (0,2,0), distance 2).
    So min distance in R^3 part is 2.
    Overall min distance is min(2sqrt(2), 2, sqrt(10)) = 2.
    Overall max norm is 2sqrt(2).
    Condition: 2 >= 2sqrt(2)? No. 2 < 2.828.
    
    So this doesn't work. We need the min distance to be at least the max norm.
    
    Let's scale the R^3 vectors. If we use 2*(D3 vectors), their norm becomes 2sqrt(2), matching the R^8 vectors.
    Then min distance in R^3 part becomes 4.
    Distance between R^8 and R^3 parts: sqrt(8 + 8) = 4.
    Min distance overall: min(2sqrt(2), 4, 4) = 2sqrt(2) ≈ 2.828.
    Max norm: 2sqrt(2).
    Condition: 2sqrt(2) >= 2sqrt(2). Holds.
    
    So the set is:
    1. 128 vectors: (v, 0, 0, 0) where v is an even-parity hypercube vertex in R^8 (coordinates ±1).
    2. 12 vectors: (0, 0, 0, 2w) where w is a D3 minimal vector (coordinates ±1,0).
    
    Total: 128 + 12 = 140.
    
    Can we do better?
    The known kissing number for R^11 is at least 252.
    How is 252 achieved?
    It might come from a different construction, possibly using the Leech lattice or other structures.
    
    Let's try a greedy approach with integer vectors to see if we can find more.
    We'll start with the 140 vectors above and try to add more.
    
    However, given the 150s budget, a greedy search might be too slow for R^11.
    
    Let's stick with the 140 for now, or try to find a better known configuration.
    
    Actually, let's reconsider the E8 construction.
    E8 has 240 minimal vectors. If we scale them to be integers, we can use:
    - 112 vectors of type (±2, ±2, 0, ..., 0) in R^8.
    - 128 vectors of type (±1, ..., ±1) with even parity in R^8.
    These are all integers.
    Norms: 2sqrt(2) for both types.
    Min distance: 
    - Between (±2,±2,0...) and (±1,...,±1): 
      Let's compute. v1=(2,2,0,...,0), v2=(1,1,1,1,1,1,1,1).
      v1.v2 = 2+2=4.
      |v1-v2|^2 = 8+8-2*4=8. Distance=2sqrt(2).
    - Between two (±2,±2,0...): min distance is 2sqrt(2) (e.g., (2,2,0..) and (2,-2,0..) differ by (0,4,0..), dist=4. Wait, (2,2,0) and (-2,2,0) differ by (4,0,0), dist=4. (2,2,0) and (2,2,0) are same. (2,2,0) and (-2,-2,0) differ by (4,4,0), dist=4sqrt(2). The min distance between distinct (±2,±2,0) vectors is 2sqrt(2) when they share one coordinate and differ in the other? No, (2,2,0) and (2,-2,0) differ by (0,4,0), dist=4. (2,2,0) and (-2,2,0) differ by (4,0,0), dist=4. (2,2,0) and (-2,-2,0) differ by (4,4,0), dist=4sqrt(2). So min distance is 4? 
      Wait, what about (2,2,0,0,0,0,0,0) and (2,0,2,0,0,0,0,0)? They are both in the set.
      v1=(2,2,0,0,0,0,0,0), v2=(2,0,2,0,0,0,0,0).
      v1.v2=4. |v1|^2=8, |v2|^2=8.
      |v1-v2|^2=8+8-2*4=8. Distance=2sqrt(2).
      Yes, so min distance is 2sqrt(2).
    - Between two (±1,...,±1) even parity: min distance is 2sqrt(2) (differ in 2 coordinates).
    
    So the 240 E8 vectors (scaled to integers) have min distance 2sqrt(2) and max norm 2sqrt(2).
    
    Now, add 3 dimensions. We can add 12 vectors of type (0,...,0, ±2, ±2, 0) in the last 3 coordinates.
    Their norm is 2sqrt(2).
    Distance between E8 vector and D3 vector: sqrt(8+8)=4.
    Min distance in D3 part: 4 (as computed before for scaled D3).
    Overall min distance: 2sqrt(2).
    Overall max norm: 2sqrt(2).
    Condition holds.
    
    Total: 240 + 12 = 252.
    
    This matches the known lower bound for k(11).
    
    Let's implement this.
    
    Steps:
    1. Generate 112 vectors of type (±2, ±2, 0, ..., 0) in R^8.
    2. Generate 128 vectors of type (±1, ..., ±1) with even number of -1s in R^8.
    3. Generate 12 vectors of type (0, ..., 0, ±2, ±2, 0) in R^3.
    4. Combine them into R^11 vectors.
    5. Return as integer array.
    """
    deadline = time.time() + 120
    
    vectors = []
    
    # 1. 112 vectors: (±2, ±2, 0, ..., 0) in R^8
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    # 2. 128 vectors: (±1, ..., ±1) with even number of -1s in R^8
    for signs in range(256):
        v = [0] * 11
        num_neg = 0
        for i in range(8):
            if (signs >> i) & 1:
                v[i] = -1
                num_neg += 1
            else:
                v[i] = 1
        if num_neg % 2 == 0:
            vectors.append(v)
    
    # 3. 12 vectors: (0, ..., 0, ±2, ±2, 0) in R^3 (last 3 coords)
    for i in range(8, 11):
        for j in range(i+1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    vec_array = np.array(vectors, dtype=np.int64)
    
    # Verify constraints (optional, for debugging)
    # norms = np.linalg.norm(vec_array, axis=1)
    # max_norm = np.max(norms)
    # dists = np.linalg.norm(vec_array[:, np.newaxis, :] - vec_array[np.newaxis, :, :], axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_dist = np.min(dists)
    # assert min_dist >= max_norm - 1e-9, f"Constraint violated: min_dist={min_dist}, max_norm={max_norm}"
    
    return vec_array
# EVOLVE-BLOCK-END
