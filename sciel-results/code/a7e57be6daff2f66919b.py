import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    We choose the complement S to be the line x = 0 in F_49^2.
    S = { (0, y) : y in F_49 }.
    N = F_49^2 \ S.
    This S has size 49.
    For any x in N, there exists a line through x disjoint from S (e.g., if x is not on x=0, 
    lines not parallel to x=0 intersect x=0 only at x, which is not in S, so they are disjoint 
    from S if we consider the intersection point. Wait, if the line intersects x=0 at x, 
    and x not in S, then L cap S = empty. Correct.)
    For any x in S, there exists a line through x such that L cap S = {x} 
    (any line not parallel to x=0 intersects x=0 only at x).
    Thus N is a valid Nikodym set.
    """
    p = 7
    # Field F_49 = F_7[alpha]
    # Point is [[x_a, x_b], [y_a, y_b]]
    
    points = []
    for xa in range(p):
        for xb in range(p):
            # Skip if x = 0 (i.e., xa=0 and xb=0)
            if xa == 0 and xb == 0:
                continue
            for ya in range(p):
                for yb in range(p):
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
