# EVOLVE-BLOCK-START
import numpy as np


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    The continuous uniform distribution on [0, 1] yields a probability of 13/48 ≈ 0.2708.
    We approximate this with a discrete uniform distribution on a fine grid.
    Using n=64 points provides a good approximation while keeping computation fast.
    """
    n = 64
    support = np.linspace(0, 1, n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64) / n
    return support, weights


# EVOLVE-BLOCK-END
