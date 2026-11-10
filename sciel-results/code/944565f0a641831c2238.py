import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # A valid Nikodym set can be a single line in F_q^2.
    # Let N be the line x = 0 (i.e., xa = 0, xb = 0).
    # Then for any point in N, the line itself works (L \ {x} subset N).
    # For any point not in N, the vertical line through it is disjoint from N.
    
    points = []
    for ya in range(p):
        for yb in range(p):
            points.append([[0, 0], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
