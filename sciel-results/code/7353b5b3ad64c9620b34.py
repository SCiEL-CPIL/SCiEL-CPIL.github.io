# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    Uses a modified Hermitian unital construction with a smaller blocking set B
    to increase the size of the excluded set S, thereby maximizing the complement
    of the Nikodym set.
    """
    p = 7

    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Reduced special y-coordinate set Y.
    # Originally Y = {(u^2 mod p, v)} for all u, v.
    # Now we restrict u to {0, 1}, so y_a in {0, 1}.
    y_special = set()
    for u in [0, 1]:
        for v in range(p):
            y_special.add(((u * u) % p, v))

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue  # excluded point (belongs to S)
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
