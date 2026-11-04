"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    vectors = []
    # Iterate over all pairs of indices (i, j)
    for i in range(D):
        for j in range(i + 1, D):
            # Select a third index k distinct from i and j
            # We can pick any k != i, j. Picking the smallest available is fine.
            # This ensures we use a consistent structure.
            k = 0
            while k in (i, j):
                k += 1
            
            # Generate all 8 sign combinations for (i, j, k)
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
