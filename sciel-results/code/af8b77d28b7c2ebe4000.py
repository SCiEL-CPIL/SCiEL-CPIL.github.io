# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    The complement S is the graph of the function f(x) = x^3, which is a scattered set.
    A scattered set S in F_q^2 is a set such that for every x in S, there exists a line
    through x that intersects S only at x.
    The maximum size of a scattered set in F_q^2 is q.
    Here q = 49, so |S| = 49, and |N| = 49^2 - 49 = 2352.
    The score is |S| = 49.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + k * a1 * b1) % p, (a0 * b1 + a1 * b0) % p)
    
    def pow3(a):
        return mul(mul(a, a), a)
        
    # Precompute x^3 for all x in F_49 to speed up lookup
    # Map (a,b) -> (a,b) to x^3
    # We can just compute on the fly since 2401 is small, but precomputation is cleaner
    # Actually, we just need to check if y == x^3.
    
    points = []
    # Iterate over all points in F_49^2
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            x3 = pow3(x)
            for ya in range(p):
                for yb in range(p):
                    y = (ya, yb)
                    # Check if (x, y) is in S (the scattered set)
                    if x3 == y:
                        continue # Exclude from N
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
