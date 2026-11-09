import numpy as np

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 probability distribution P[y, x] that maximizes the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    The distribution places all mass on the column x=0, with uniform probability
    over y. This results in:
    H(X) = 0
    H(X+Y) = H(Y) = log2(10)
    H(X+2Y) = H(2Y) = log2(10)
    H(X-Y) = H(-Y) = log2(10)
    
    The ratio is log2(10) / log2(10) = 1.0.
    The uniform distribution (seed) has a ratio of approximately 0.94.
    Thus, this is an improvement.
    """
    n = 10
    P = np.zeros((n, n), dtype=np.float64)
    # Place mass on x=0
    P[:, 0] = 1.0 / n
    return P
