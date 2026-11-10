# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2. Uses the Hermitian unital construction
    with a refined exclusion criterion to maximize the complement size.
    
    The unital U = {(x,y) : Trace(x) + Norm(y) = 0} has 50 points per x-fiber
    (actually |U| = q + 1 = 50 total... no, for q=49, unital has q+1=50 points).
    Wait, a unital in PG(2,q) has q+1 points. But here we're in affine F_q^2.
    
    Actually the equation Trace(x) + Norm(y) = 0 in F_49^2: for each x (49 choices),
    we need Norm(y) = -Trace(x). The number of y with Norm(y) = c is either
    q-1=48 (if c is a norm) or 0 or something else. Let me just count.
    
    For our purposes, we exclude a subset of the unital points to form the
    complement, and N = F_49^2 \\ excluded.
    """
    p = 7
    k = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)
    
    # Quadratic residues mod 7: {0, 1, 2, 4}
    QR = {0, 1, 2, 4}
    
    # Build the excluded set: points in the unital with y_a in QR
    # This gives a larger excluded set than using non-QR (per knowledge)
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    if in_unital and ya in QR:
                        continue  # excluded
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
