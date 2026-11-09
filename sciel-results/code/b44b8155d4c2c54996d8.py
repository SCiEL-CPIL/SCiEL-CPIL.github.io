# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where the complement S is a line
    y = m*x for some m in F_49. A line has q = 49 points.
    
    For each point x on the line S, we need a line through x (with direction
    v = (1, m') for some m' in F_49, or v = (0,1)) such that all other points
    on that line are in N (i.e., not on S).
    
    If S is the line y = m0*x, then for a point x0 = (t, m0*t) on S,
    consider the vertical line through x0: {(t, y) : y in F_49}.
    The only point of S on this vertical line is x0 itself (since S intersects
    any vertical line at exactly one point). So all other 48 points on the
    vertical line are in N. This satisfies the Nikodym condition.
    
    Thus N = F_49^2 \\ S has size 2401 - 49 = 2352.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3. Element (a, b) = a + b*alpha.
    Returns N as integer array of shape (k, 2, 2).
    """
    p = 7
    k = 3  # alpha^2 = 3 in F_7
    
    # Choose S to be the line y = 0 (the x-axis)
    # S = {(x, (0,0)) : x in F_49}
    # This is 49 points.
    
    # N = all points except those on y = 0
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Point is ((xa, xb), (ya, yb))
                    # y = (ya, yb), we want y != (0, 0)
                    if ya != 0 or yb != 0:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
