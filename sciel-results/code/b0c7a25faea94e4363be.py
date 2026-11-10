import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    
    # Generate U points
    points_U = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        points_U.append((xa, xb, ya, yb))
                        
    # Pick 2 points to keep in N (R)
    # Any 2 distinct points work
    R = {points_U[0], points_U[1]}
    
    points_N = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    in_U = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    if in_U and pt not in R:
                        continue # Excluded (in S)
                    points_N.append([[xa, xb], [ya, yb]])
                    
    return np.array(points_N, dtype=np.int64)
