# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    We construct the complement C as a conic U defined by Tr(x) + Nm(y) = 0.
    The size of C is 343.
    The Nikodym set N is F_49^2 \ C.
    """
    p = 7
    # k is the quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if (x, y) is in U: 2*xa + ya^2 - yb^2*k == 0 mod p
                    val = (2 * xa + ya * ya - yb * yb * k) % p
                    if val == 0:
                        continue  # This point is in C (complement)
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
