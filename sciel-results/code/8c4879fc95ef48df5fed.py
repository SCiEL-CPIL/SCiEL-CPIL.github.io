# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where the complement S is an oval
    (conic) x^2 + (1+alpha)*y^2 = 1, which has q+1 = 50 points.
    For each point s in S, the tangent line at s intersects S only at s,
    satisfying the Nikodym condition.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3. Element (a, b) = a + b*alpha.
    Returns N = F_49^2 \\ S as integer array of shape (k, 2, 2).
    """
    p = 7
    k = 3  # alpha^2 = 3 in F_7
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = k
    def add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
    def mul(x, y):
        a, b = x
        c, d = y
        # (a + b*alpha)(c + d*alpha) = (ac + bd*k) + (ad + bc)*alpha
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
    
    def neg(x):
        return ((-x[0]) % p, (-x[1]) % p)
    
    def inv(x):
        # x = (a, b), x^{-1} = (a, -b) / (a^2 + b^2*k)
        a, b = x
        denom = (a * a + b * b * k) % p
        if denom == 0:
            raise ZeroDivisionError("Cannot invert zero")
        inv_denom = pow(denom, p - 2, p)
        return ((a * inv_denom) % p, (-b * inv_denom) % p)
    
    # Enumerate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # c = 1 + alpha = (1, 1)
    c = (1, 1)
    one = (1, 0)
    
    # Find all points (x, y) in F_49^2 such that x^2 + c*y^2 = 1
    S = set()
    for x in field_elements:
        x2 = mul(x, x)
        for y in field_elements:
            y2 = mul(y, y)
            cy2 = mul(c, y2)
            lhs = add(x2, cy2)
            if lhs == one:
                S.add((x, y))
    
    # N = all points not in S
    # Pre-allocate array for max size
    max_points = p * p * p * p
    N_points = np.zeros((max_points, 2, 2), dtype=np.int64)
    count = 0
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    if (x, y) not in S:
                        N_points[count, 0, 0] = xa
                        N_points[count, 0, 1] = xb
                        N_points[count, 1, 0] = ya
                        N_points[count, 1, 1] = yb
                        count += 1
                        
    return N_points[:count]


# EVOLVE-BLOCK-END
