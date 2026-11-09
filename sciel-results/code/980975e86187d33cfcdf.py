import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a 2-point discrete probability measure on {0, 1} that maximizes
    P[X1 + X2 + X3 < 2*X4].
    
    The optimal probability mass p on the point 1 is the root of 8p^3 - 9p^2 + 1 = 0,
    which is p = (1 + sqrt(33)) / 16.
    """
    p = (1.0 + np.sqrt(33.0)) / 16.0
    support = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([1.0 - p, p], dtype=np.float64)
    return support, weights
