# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by taking the complement of the Hermitian unital U.
    U = {(x, y) : Trace(x) + Norm(y) = 0}
    This set S = U has the property that every point in S has a unique tangent line
    that intersects S only at that point. Thus, N = F_49^2 \ S is a valid Nikodym set.
    Size of S is 343, so Size of N is 2058.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    points = []
    
    # Iterate over all points in F_49^2
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Calculate Trace(x) = 2 * xa
                    trace_x = (2 * xa) % p
                    
                    # Calculate Norm(y) = ya^2 - k * yb^2
                    norm_y = (ya * ya - k * yb * yb) % p
                    
                    # Check if point is in U: Trace(x) + Norm(y) == 0
                    if (trace_x + norm_y) % p == 0:
                        # This point is in S (excluded from N)
                        continue
                    else:
                        # This point is in N
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
