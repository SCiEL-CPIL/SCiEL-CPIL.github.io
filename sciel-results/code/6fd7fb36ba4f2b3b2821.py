import numpy as np
import time
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    
    # E8 lattice minimal vectors have norm^2 = 2
    # There are 240 minimal vectors in E8:
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) - 2^3 * 8 * 2 = 256? No.
    # Actually: 
    # Type A: (±1, ±1, 0, 0, 0, 0, 0, 0) with even number of -1's? No.
    # The 240 minimal vectors of E8 are:
    # 112 vectors of type (±1, ±1, 0, 0, 0, 0, 0, 0) [2^3 * 8 * 2 = ... let me count]
    #   Choose 2 positions out of 8: C(8,2) = 28
    #   For each pair, 4 sign combinations: (+1,+1), (+1,-1), (-1,+1), (-1,-1)
    #   But E8 requires even parity? No, E8 is {x in Z^8 : sum x_i = 0 mod 2} ∪ {x in (Z+1/2)^8 : ...}
    #   Wait, the standard E8 lattice is {x in Z^8 : sum x_i ≡ 0 (mod 2)}
    #   So for (±1, ±1, 0, ..., 0): sum = ±1 ± 1, which is even (0 or ±2). So all 4 sign combos work.
    #   Count: C(8,2) * 4 = 28 * 4 = 112
    # 128 vectors of type (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of -1's
    #   Count: 2^7 = 128 (half of all 2^8 sign combos)
    # Total: 112 + 128 = 240. Correct.
    
    # But we need integer vectors. The half-integer ones aren't integers.
    # So from E8, only the 112 integer minimal vectors work directly.
    
    # However, we can embed E8 in R^11 and use the 240 vectors.
    # But the half-integer vectors need to be scaled. If we scale by 2, they become integers.
    # Let's use a different approach: use the D11 lattice or construct directly.
    
    # Actually, let me think about what gives us the most points.
    # The kissing number in 11D is known to be 240 (or close to it).
    # Let's try to construct 240 integer points.
    
    # Strategy: Use vectors of small norm.
    # If we use vectors with norm^2 = 2 (i.e., two ±1's and rest 0):
    #   Number of such vectors in Z^11: C(11,2) * 4 = 55 * 4 = 220
    #   Min distance between any two: need to check.
    #   Two vectors sharing one coordinate: e.g., (1,1,0,...) and (1,-1,0,...) -> dist^2 = 4
    #   Two vectors sharing no coordinates: e.g., (1,1,0,...) and (0,0,1,1,...) -> dist^2 = 4
    #   Two vectors sharing both coordinates: same vector, not distinct.
    #   So min dist^2 = 4, max norm^2 = 2. 4 >= 2. Valid!
    #   But k = 220.
    
    # Can we do better? Add vectors of norm^2 = 4 (four ±1's)?
    #   Number: C(11,4) * 16 = 330 * 16 = 5280. But they may not be compatible.
    #   A norm^2=4 vector and a norm^2=2 vector:
    #     If the norm^2=2 vector's support is subset of norm^2=4's support:
    #       e.g., (1,1,0,...) and (1,1,1,1,0,...): dot = 2, dist^2 = 2+4-4 = 2. 
    #       But we need dist^2 >= max(norm^2) = 4. So 2 < 4. Not compatible.
    #     So can't mix norm^2=2 and norm^2=4 easily.
    
    # What about using norm^2 = 4 vectors only?
    #   Need min dist^2 >= 4.
    #   Two norm^2=4 vectors: dist^2 = 4+4-2*dot = 8-2*dot.
    #   Need 8-2*dot >= 4 => dot <= 2.
    #   Dot product of two vectors with four ±1's each:
    #     If they share k coordinates with same sign and l with opposite: dot = k - l
    #     k + l <= min(4,4) = 4 (if supports overlap in k+l positions)
    #     Actually, if supports intersect in m positions, then k+l = m, and k-l = dot.
    #     m can be 0,1,2,3,4.
    #     If m=4 (same support): dot = ±4. If same signs: dot=4, dist^2=0. Bad.
    #     If m=3: dot can be -3,-1,1,3. If dot=3: dist^2=8-6=2 < 4. Bad.
    #     So need careful selection.
    
    # Let's stick with the 220 norm^2=2 vectors and see if we can augment.
    
    # Generate all 220 vectors of type (±1, ±1, 0, ..., 0)
    vectors = []
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    k = len(vectors)
    print(f"Generated {k} vectors of norm^2=2")
    
    # Verify: min dist^2 >= max norm^2
    norms_sq = np.sum(vectors**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute all pairwise distances
    # k=220, so 220*219/2 = 24090 pairs. Fast.
    min_dist_sq = np.inf
    for i in range(k):
        for j in range(i+1, k):
            diff = vectors[i] - vectors[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    print(f"Max norm^2 = {max_norm_sq}, Min dist^2 = {min_dist_sq}")
    assert min_dist_sq >= max_norm_sq, f"Invalid: min_dist_sq={min_dist_sq} < max_norm_sq={max_norm_sq}"
    
    # Now try to add more vectors. 
    # We can try vectors of type (±2, 0, ..., 0) - norm^2 = 4.
    #   Number: 11 * 2 = 22.
    #   Check compatibility with norm^2=2 vectors:
    #     (2,0,...) and (1,1,0,...): dot = 2, dist^2 = 4+2-4 = 2 < 4. Bad.
    #     (2,0,...) and (0,1,1,...): dot = 0, dist^2 = 4+2-0 = 6 >= 4. OK.
    #   So (2,0,...) is compatible with norm^2=2 vectors that don't have a ±1 in position 0.
    #   But incompatible with those that do.
    #   So we can't add (2,0,...) if we already have vectors with ±1 in position 0.
    
    # Alternative: try to find a larger set using a different construction.
    # Let's try the Leech lattice approach or use known results.
    
    # Actually, the kissing number in 11D is 240. Let me try to achieve that.
    # One known construction uses the D11 lattice.
    
    # D_n = {x in Z^n : sum x_i ≡ 0 (mod 2)}
    # Minimal vectors of D_11: (±1, ±1, 0, ..., 0) with even sum.
    #   Sum = ±1 ± 1 = 0 or ±2, all even. So all 220 are in D_11.
    #   But D_11 has more minimal vectors? No, minimal vectors are exactly these.
    
    # The kissing number of D_11 is 220. But the kissing number of R^11 is 240.
    # The extra 20 come from a different lattice.
    
    # For now, return the 220 vectors. This is a valid configuration.
    # We can try to improve later.
    
    return vectors
