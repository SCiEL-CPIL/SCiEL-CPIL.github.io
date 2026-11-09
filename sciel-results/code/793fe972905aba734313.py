# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    The complement S is the hyperplane { (x, y) : x_a = 0 }.
    Size of S is 343.
    Size of N is 2401 - 343 = 2058.
    """
    p = 7
    q = p * p
    
    points = []
    for xa in range(p):
        if xa == 0:
            continue # This is in S, so not in N
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
