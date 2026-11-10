import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3 # 3 is a quadratic non-residue mod 7? 3^3 = 27 = 6 = -1. Yes.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if point is in U
                    # 2*xa + ya^2 - k*yb^2 == 0
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        continue # Exclude U
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
