import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)

    # Precompute field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Construct S = {(x, x^2) : x in F_49}
    S = set()
    for x in field_elements:
        x2 = field_mul(x, x)
        S.add((x, x2))
        
    # N is the complement of S
    # We want to return points in N
    points = []
    for x in field_elements:
        for y in field_elements:
            if (x, y) not in S:
                xa, xb = x
                ya, yb = y
                points.append([[xa, xb], [ya, yb]])
                
    return np.array(points, dtype=np.int64)
