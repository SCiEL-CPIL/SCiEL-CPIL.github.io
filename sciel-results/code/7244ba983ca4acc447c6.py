# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Returns a valid integer kissing configuration in dimension 11.
    Uses all minimal vectors of the D_11 lattice (weight-2 vectors).
    Number of points: 4 * C(11, 2) = 220.
    Max squared norm: 2.
    Min squared pairwise distance: 2.
    Condition: 2 >= 2 holds.
    """
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 540
    
    d = 11
    pts = []
    
    # Generate all vectors of form +/- e_i +/- e_j
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
                    
    # Convert to array
    arr = np.array(pts, dtype=np.int64)
    
    # Apply rounding and casting as per contract
    arr = np.around(arr).astype(np.int64)
    
    return arr

# EVOLVE-BLOCK-END
