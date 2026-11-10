# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49.
    Uses a geometric construction based on a Hermitian unital.
    """
    p = 7
    q = p * p
    # alpha^2 = 3 (smallest quadratic non-residue mod 7 is 3? 
    # 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1. 
    # Non-residues: 3, 5, 6. Smallest is 3.)
    k = 3
    
    # Field operations for F_49
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # alpha^2 = k
        a0, a1 = a
        b0, b1 = b
        r0 = (a0 * b0 + a1 * b1 * k) % p
        r1 = (a0 * b1 + a1 * b0) % p
        return (r0, r1)
        
    def neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
        
    def sub(a, b):
        return add(a, neg(b))
        
    # Precompute all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Define the Unital U: Trace(x) + Norm(y) = 0? 
    # The seed used: 2*xa + ya*ya - yb*yb*k == 0
    # This is a specific curve. Let's stick to it as it scored well.
    
    # Y set: y_a is a square mod p
    squares = set((u * u) % p for u in range(p))
    
    # Construct S (complement)
    S = set()
    for x, y in points:
        xa, xb = x
        ya, yb = y
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S.add(((xa, xb), (ya, yb)))
            
    # N is the rest
    N = [pt for pt in points if pt not in S]
    
    # Convert to array
    arr = np.array([[[x[0], x[1]], [y[0], y[1]]] for x, y in N], dtype=np.int64)
    
    return arr

# EVOLVE-BLOCK-END
