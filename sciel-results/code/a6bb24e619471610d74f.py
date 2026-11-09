import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Generate all points
    # Point is [x_a, x_b, y_a, y_b]
    points = np.array([(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)], dtype=np.int64)
    
    # Define S: the line y = 0
    # y_a = 0, y_b = 0
    # x_a, x_b vary
    S_indices = []
    for i, pt in enumerate(points):
        if pt[2] == 0 and pt[3] == 0:
            S_indices.append(i)
            
    # N is the complement of S
    N_indices = [i for i in range(len(points)) if i not in S_indices]
    
    # Return points in N
    N_points = points[N_indices]
    
    return N_points
