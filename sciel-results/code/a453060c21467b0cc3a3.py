# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array.
    """
    p = 7
    k = 3  # alpha^2 = 3

    # Squares mod 7: {0, 1, 2, 4}
    squares = set((u * u) % p for u in range(p))
    
    # y_special: y_a is a square mod p
    y_special = set()
    for u in range(p):
        sq = (u * u) % p
        for v in range(p):
            y_special.add((sq, v))

    points = []
    # Iterate through all 2401 points
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if excluded (in unital and y not special)
                    # Unital condition: 2*xa + ya^2 - yb^2*k == 0 mod p
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if (ya, yb) not in y_special:
                            continue  # Excluded
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
