# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    The set N is the complement of the Hermitian unital U.
    U = {(x, y) : Trace(x) + Norm(y) = 0}.
    Since every point on the unital has a tangent line that intersects U only at that point,
    U is a valid complement for a Nikodym set.
    The size of U is q * p = 49 * 7 = 343.
    The size of N is 2401 - 343 = 2058.
    The complement size is 343, which is larger than the previous 147.
    """
    p = 7
    q = p * p

    # Smallest quadratic non-residue mod p
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
                    # Check if (x, y) is in the unital U
                    # Trace(x) = 2 * xa
                    # Norm(y) = ya^2 - k * yb^2
                    # Condition: 2 * xa + ya^2 - k * yb^2 == 0 (mod p)
                    if (2 * xa + ya * ya - k * yb * yb) % p != 0:
                        # Not in unital, so it is in N
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
