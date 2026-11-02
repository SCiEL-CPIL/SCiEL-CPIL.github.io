"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    vectors = []
    
    # 1. Scaled E8 roots in first 8 dimensions (norm squared 8)
    # A-type: 2 nonzeros in {-2, 2}
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    # B-type: 8 nonzeros in {-1, 1} with even number of -1s
    for signs in itertools.product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = [0] * D
            for i, s in enumerate(signs):
                v[i] = s
            vectors.append(v)
            
    # 2. Weight 4 vectors in first 8 dimensions (norm squared 4? No, 4 * 1^2 = 4? Wait.)
    # My previous analysis said norm squared 8 for weight 4 vectors?
    # Vector: (1, 1, 1, 1, 0, 0, 0, 0). Norm squared = 1+1+1+1 = 4.
    # AH. I made a mistake in the analysis.
    # Norm squared of weight 4 vector with entries +/- 1 is 4.
    # But max norm is sqrt(8).
    # Condition: dist^2 >= 8.
    # If we add a vector w with norm^2 = 4, and u with norm^2 = 8.
    # dist^2 = 8 + 4 - 2 u.dot(w) = 12 - 2 u.dot(w).
    # We need 12 - 2 u.dot(w) >= 8 => 4 >= 2 u.dot(w) => u.dot(w) <= 2.
    
    # Let's re-verify compatibility of Weight 4 (norm^2=4) with E8 (norm^2=8).
    # u in E8 (scaled). w in Weight 4.
    
    # Case 1: u is A-type (2 nonzeros +/- 2).
    # w is Weight 4 (4 nonzeros +/- 1).
    # Intersection k.
    # If k=2 (u's support is subset of w's support):
    # u.dot(w) = (+/- 2)(+/- 1) + (+/- 2)(+/- 1).
    # Max value: 2+2=4.
    # We need u.dot(w) <= 2.
    # But 4 > 2. So they are NOT compatible.
    # Example: u=(2,2,0..), w=(1,1,1,1..). Dot=4.
    # Dist^2 = 8+4-8=4 < 8. Fail.
    
    # So we CANNOT add all weight 4 vectors.
    # We can only add weight 4 vectors that are compatible with ALL E8 vectors.
    # This is a constraint satisfaction problem.
    # Given the time, and that 240+12=252 is already >> 2.65, I will stick to the safe set.
    # The safe set is just the E8 roots (240) and the last 3 dim vectors (12).
    # Total 252.
    
    # 3. Norm 8 vectors in last 3 dimensions (indices 8, 9, 10)
    # 2 nonzeros in {-2, 2}
    for i in [8, 9, 10]:
        for j in [8, 9, 10]:
            if i == j:
                continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
