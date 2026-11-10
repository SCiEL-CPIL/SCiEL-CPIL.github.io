import numpy as np
from scipy.optimize import differential_evolution

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # Optimal two-point distribution for maximizing P(X1+X2+X3 < 2*X4)
    # Support: [0, A] with A > 0
    # Weights: [p, 1-p] where p = (15 - sqrt(33))/16
    # P = p^2 * (1-p) * (3 - 2*p)
    
    p = (15.0 - np.sqrt(33.0)) / 16.0
    A = 1.0  # Any positive value works; choose 1.0 for simplicity
    
    support = np.array([0.0, A], dtype=np.float64)
    weights = np.array([p, 1.0 - p], dtype=np.float64)
    
    # Ensure validity
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        # Fallback to uniform on [0,1]
        support = np.array([0.0, 1.0], dtype=np.float64)
        weights = np.array([0.5, 0.5], dtype=np.float64)
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        weights = weights / w_sum
    
    return support, weights
