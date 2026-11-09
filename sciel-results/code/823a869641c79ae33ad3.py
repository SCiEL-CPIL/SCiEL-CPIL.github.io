# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by taking the complement of a Hermitian unital.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    The Hermitian unital U is defined by Tr(x) + Norm(y) = 0, where:
    Tr(a + b*alpha) = 2a
    Norm(a + b*alpha) = a^2 - 3b^2
    
    The set N = F_49^2 \ U is a Nikodym set.
    Size of U is 344.
    Size of N is 2401 - 344 = 2057.
    Complement size is 344.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Precompute Norm and Trace for all elements
    # Element represented by (a, b)
    trace = np.zeros((p, p), dtype=np.int64)
    norm = np.zeros((p, p), dtype=np.int64)
    for a in range(p):
        for b in range(p):
            trace[a, b] = (2 * a) % p
            norm[a, b] = (a * a - b * b * k) % p
            
    points = []
    for xa in range(p):
        for xb in range(p):
            tx = trace[xa, xb]
            for ya in range(p):
                for yb in range(p):
                    ny = norm[ya, yb]
                    # Check if (x, y) is in U: Tr(x) + Norm(y) == 0
                    if (tx + ny) % p == 0:
                        continue  # This point is in U, so it is NOT in N
                    points.append([[xa, xb], [ya, yb]])
                    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
