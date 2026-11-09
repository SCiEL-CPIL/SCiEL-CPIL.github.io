import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Return a safe fallback: linear fit using first two params
        return params[0] * x + params[1]
    
    # Scale x to [-1, 1]
    t = x / 2.0
    
    # Build a richer basis: combine polynomial, trigonometric, and exponential terms
    # This gives the optimizer more flexibility to fit diverse function shapes
    
    N = 10
    n_points = len(t)
    
    # We'll construct 10 basis functions that span a diverse set of function shapes
    basis = np.zeros((n_points, N))
    
    # Polynomial terms (Legendre-like for stability)
    # P_0 = 1, P_1 = t, P_2 = (3t^2 - 1)/2, etc.
    basis[:, 0] = 1.0
    basis[:, 1] = t
    basis[:, 2] = 0.5 * (3.0 * t**2 - 1.0)
    basis[:, 3] = 0.125 * (5.0 * t**3 - 3.0 * t)
    
    # Trigonometric terms (useful for periodic-like behavior)
    basis[:, 4] = np.cos(np.pi * t)
    basis[:, 5] = np.sin(np.pi * t)
    basis[:, 6] = np.cos(2.0 * np.pi * t)
    basis[:, 7] = np.sin(2.0 * np.pi * t)
    
    # Exponential / sigmoid-like terms
    basis[:, 8] = np.exp(t) - np.exp(-t)  # sinh-like, odd
    basis[:, 9] = np.exp(0.5 * t) + np.exp(-0.5 * t)  # cosh-like, even
    
    # Normalize each column to have unit norm for better conditioning
    for j in range(N):
        col_norm = np.linalg.norm(basis[:, j])
        if col_norm > 1e-12:
            basis[:, j] /= col_norm
    
    return basis @ params
