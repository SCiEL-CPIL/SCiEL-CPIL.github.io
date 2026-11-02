"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    vectors = []
    
    # Generate all vectors with exactly 4 non-zero entries of +1 or -1
    # Count: C(11, 4) * 2^4 = 330 * 16 = 5280
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            if time.time() > deadline:
                break
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
        if time.time() > deadline:
            break
            
    if not vectors:
        # Fallback to seed if something goes wrong or time is extremely tight
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)

    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
