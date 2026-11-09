"""Symbolic regression with 1 input(s) using a diverse elementary function basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    # Ensure x is in a safe range for operations, though [0.1, 6] is safe for exp, sin, cos, 1/x
    # We construct the basis matrix of shape (N, 10)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0)
        
    basis = np.zeros((n, 10), dtype=np.float64)
    
    # 1. Constant
    basis[:, 0] = 1.0
    # 2. Linear
    basis[:, 1] = x
    # 3. Quadratic
    basis[:, 2] = x**2
    # 4. Cubic
    basis[:, 3] = x**3
    # 5. Quartic
    basis[:, 4] = x**4
    # 6. Exponential
    basis[:, 5] = np.exp(x)
    # 7. Negative Exponential
    basis[:, 6] = np.exp(-x)
    # 8. Sine
    basis[:, 7] = np.sin(x)
    # 9. Cosine
    basis[:, 8] = np.cos(x)
    # 10. Inverse
    basis[:, 9] = 1.0 / x
    
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
