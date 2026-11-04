"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate all vectors with exactly 3 non-zero entries (+/- 1)
    # There are C(11, 3) * 2^3 = 165 * 8 = 1320 such vectors
    vectors = []
    for indices in itertools.combinations(range(D), 3):
        i, j, k = indices
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(v)
    
    # Greedy selection
    selected = []
    # To make the greedy selection more effective, we can shuffle or sort.
    # Given the symmetry, any order should work similarly, but let's keep it simple.
    
    for v in vectors:
        if time.time() > deadline:
            break
            
        # Check compatibility with all selected vectors
        compatible = True
        for u in selected:
            diff = v - u
            dist_sq = np.dot(diff, diff)
            if dist_sq < 3:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
            
    if not selected:
        # Fallback to seed if something went wrong
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
