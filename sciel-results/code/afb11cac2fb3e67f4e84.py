import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    points_in_N = []
    for xa in range(p):
        for xb in range(p):
            # Check if x is zero in F_q
            if xa == 0 and xb == 0:
                continue
            for ya in range(p):
                for yb in range(p):
                    points_in_N.append([[xa, xb], [ya, yb]])
                    
    return np.array(points_in_N, dtype=np.int64)
