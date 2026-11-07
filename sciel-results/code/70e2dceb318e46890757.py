"""Symbolic regression with 2 inputs using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Time tracking for safety
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Conservative margin before 120s kill

def _chebyshev_poly(x, degree):
    """
    Compute Chebyshev polynomials of the first kind up to 'degree' for input x.
    Returns an array of shape (len(x), degree+1).
    """
    n = len(x)
    T = np.zeros((n, degree + 1))
    if degree >= 0:
        T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x
    for k in range(2, degree + 1):
        T[:, k] = 2.0 * x * T[:, k - 1] - T[:, k - 2]
    return T

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Full cubic polynomial in x and y using Chebyshev basis.
    The basis consists of 10 terms: T_i(x) * T_j(y) where i + j <= 3.
    params: array of length 10.
    """
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a safe default if we are out of time (though this shouldn't happen for this simple model)
        return np.zeros_like(x)

    n_points = len(x)
    
    # Compute Chebyshev polynomials up to degree 3 for x and y
    # T_x shape: (n_points, 4) -> columns are T0(x), T1(x), T2(x), T3(x)
    # T_y shape: (n_points, 4) -> columns are T0(y), T1(y), T2(y), T3(y)
    T_x = _chebyshev_poly(x, 3)
    T_y = _chebyshev_poly(y, 3)
    
    # Construct the design matrix Phi with 10 columns
    # We need all pairs (i, j) such that i + j <= 3
    # Indices:
    # 0: T0(x)*T0(y)
    # 1: T1(x)*T0(y)
    # 2: T0(x)*T1(y)
    # 3: T2(x)*T0(y)
    # 4: T1(x)*T1(y)
    # 5: T0(x)*T2(y)
    # 6: T3(x)*T0(y)
    # 7: T2(x)*T1(y)
    # 8: T1(x)*T2(y)
    # 9: T0(x)*T3(y)
    
    Phi = np.empty((n_points, 10), dtype=np.float64)
    Phi[:, 0] = T_x[:, 0] * T_y[:, 0]
    Phi[:, 1] = T_x[:, 1] * T_y[:, 0]
    Phi[:, 2] = T_x[:, 0] * T_y[:, 1]
    Phi[:, 3] = T_x[:, 2] * T_y[:, 0]
    Phi[:, 4] = T_x[:, 1] * T_y[:, 1]
    Phi[:, 5] = T_x[:, 0] * T_y[:, 2]
    Phi[:, 6] = T_x[:, 3] * T_y[:, 0]
    Phi[:, 7] = T_x[:, 2] * T_y[:, 1]
    Phi[:, 8] = T_x[:, 1] * T_y[:, 2]
    Phi[:, 9] = T_x[:, 0] * T_y[:, 3]
    
    # Linear combination of basis functions
    pred = Phi @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        return np.zeros(n_points)
        
    return pred
# EVOLVE-BLOCK-END
