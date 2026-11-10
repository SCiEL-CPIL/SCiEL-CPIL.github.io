import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field arithmetic
    def add_f49(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    def mul_f49(u, v):
        a, b = u
        c, d = v
        return ((a*c + b*d*k) % p, (a*d + b*c) % p)

    # Generate points
    points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    idx = len(points)
                    points.append(pt)
                    point_to_idx[pt] = idx

    # Define S
    S_mask = np.zeros(len(points), dtype=bool)
    
    # L: y = 0 (ya=0, yb=0)
    for xa in range(p):
        for xb in range(p):
            pt = ((xa, xb), (0, 0))
            S_mask[point_to_idx[pt]] = True
            
    # P: subset of y = 1 (ya=1, yb=0) of size q-1
    # Exclude one point, e.g., (0,0) in x-coords? No, x is in F49.
    # Exclude (0,0) from the line y=1.
    excluded_pt = ((0, 0), (1, 0))
    for xa in range(p):
        for xb in range(p):
            pt = ((xa, xb), (1, 0))
            if pt != excluded_pt:
                S_mask[point_to_idx[pt]] = True

    # Construct N
    N_points = []
    for pt, idx in point_to_idx.items():
        if not S_mask[idx]:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)
