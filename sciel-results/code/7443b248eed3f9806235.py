"""Symbolic regression with 1 input; Hybrid global polynomial + local RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: global Chebyshev polynomials (low-order) + local RBF kernels.
    
    The 10 parameters are split:
      - First 5: weights for Chebyshev polynomials T_0..T_4 (global trend)
      - Last 5: weights for 5 RBF kernels (local features)
    
    This combination allows the model to capture both global structure
    (via the polynomial part) and local variations (via the RBF part),
    improving conditioning and fit quality.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
    
    # --- Global part: Chebyshev polynomials T_0..T_4 ---
    x_min, x_max = 0.1, 6.0
    # Map x from [0.1, 6] to [-1, 1]
    z = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    z = np.clip(z, -1.0, 1.0)
    
    n_cheb = 5
    T = np.zeros((n_cheb, len(x)))
    T[0, :] = 1.0
    if n_cheb > 1:
        T[1, :] = z
    for n in range(2, n_cheb):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
    
    # Chebyshev contribution: shape (len(x),)
    y_cheb = T @ params[:n_cheb]
    
    # --- Local part: RBF kernels ---
    n_rbf = 5
    # 5 evenly-spaced centers across [0.1, 6]
    centers = np.linspace(x_min + (x_max - x_min) / (2 * n_rbf),
                          x_max - (x_max - x_min) / (2 * n_rbf), n_rbf)
    
    # Moderate sigma for smooth local interpolation
    sigma = 1.2
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis_rbf = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (N, n_rbf)
    
    y_rbf = basis_rbf @ params[n_cheb:]
    
    # Total prediction
    y = y_cheb + y_rbf
    
    return y
# EVOLVE-BLOCK-END
