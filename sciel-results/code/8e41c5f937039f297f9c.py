"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    vectors = []
    # Generate all vectors with exactly 4 non-zero entries, each being +/- 1
    # Norm is sqrt(4) = 2.
    # Min distance is 2 (e.g., same support, one sign flip).
    # Condition: min_dist >= max_norm => 2 >= 2.
    
    for combo in itertools.combinations(range(D), 4):
        # 16 sign combinations for 4 positions
        for mask in range(16):
            v = [0] * D
            for i, idx in enumerate(combo):
                # bit i of mask determines sign
                if (mask >> i) & 1:
                    v[idx] = 1
                else:
                    v[idx] = -1
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
