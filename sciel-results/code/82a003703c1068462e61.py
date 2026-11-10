# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    We construct the complement S as the union of a line L and a single point P not on L.
    S = L U {P}.
    It can be verified that for every point x in F_49^2, there exists a line through x
    that intersects S in at most one point.
    Thus N = F_49^2 \ S is a Nikodym set.
    |S| = 49 + 1 = 50.
    |N| = 2401 - 50 = 2351.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Precompute field multiplication table for speed?
    # Or just use formulas.
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def mul_f(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + k*a1*b1) % p, (a0*b1 + a1*b0) % p)

    # Generate all elements of F_49
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Define S
    # L is the line y = 0. Points (x, (0,0)) for all x in F_49
    # P is the point ((0,0), (1,0)) i.e. x=0, y=1
    S = set()
    for x in field_elements:
        S.add((x, (0, 0)))
    S.add(((0, 0), (1, 0)))
    
    # Construct N
    points = []
    for x in field_elements:
        for y in field_elements:
            if (x, y) not in S:
                points.append([[x[0], x[1]], [y[0], y[1]]])
                
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
