import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    p = (1 + np.sqrt(33)) / 16
    w0 = 1 - p
    w1 = p
    support = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([w0, w1], dtype=np.float64)
    return support, weights
