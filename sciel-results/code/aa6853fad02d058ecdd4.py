import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)
    
    # We want N to be the complement of a line S.
    # Let S be the line y = 0.
    # S = { (x, 0) : x in F_49 }
    # N = F_49^2 \ S
    
    # Verify validity:
    # 1. For x in S (y=0):
    #    We need a line through x such that other points are in N.
    #    Pick line with slope m=1 (direction (1,1)).
    #    Line: (x0 + t, 0 + t) = (x0+t, t).
    #    For t=0, point is (x0, 0) = x.
    #    For t != 0, y-coord is t != 0.
    #    So all other points have y != 0, thus are in N.
    #    Valid.
    # 2. For x not in S (y != 0):
    #    We need a line through x such that other points are in N.
    #    Pick horizontal line y = x_y.
    #    Line: (x0 + t, x_y).
    #    All points have y = x_y != 0.
    #    So all points are in N.
    #    Thus L \ {x} is in N.
    #    Valid.
    
    # So N = { (x, y) : y != 0 } is a valid Nikodym set.
    # Size of N = 49 * 48 = 2352.
    # Complement size = 49.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            for ya in range(p):
                for yb in range(p):
                    y = (ya, yb)
                    # Check if y == 0
                    if not (ya == 0 and yb == 0):
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
