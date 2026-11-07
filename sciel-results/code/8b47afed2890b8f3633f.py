"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis in scaled (x, v) for the nonlinear oscillator.
    
    Uses tensor-product Chebyshev polynomials up to order 2 in each variable,
    giving 9 basis terms (T00, T10, T01, T20, T11, T02, T21, T12, T22).
    The 10th parameter scales the overall output for better conditioning.
    
    params[0:9] : coefficients for Chebyshev basis terms
    params[9]   : global scale factor
    """
    # Scale inputs to [-1, 1] for Chebyshev polynomials
    # x range ~ [-0.57, 0.53], v range ~ [-0.18, 0.28]
    X = x / 0.55
    V = v / 0.25
    
    # Clip to [-1, 1] to stay in Chebyshev domain (slight extrapolation ok)
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Chebyshev polynomials: T0=1, T1=t, T2=2t^2-1
    T0 = np.ones_like(X)
    T1X = X
    T2X = 2.0 * X * X - 1.0
    T1V = V
    T2V = 2.0 * V * V - 1.0
    
    # Tensor-product basis (9 terms for orders 0,1,2 in each variable)
    basis = (
        params[0] * T0 * T0 +
        params[1] * T1X * T0 +
        params[2] * T0 * T1V +
        params[3] * T2X * T0 +
        params[4] * T1X * T1V +
        params[5] * T0 * T2V +
        params[6] * T2X * T1V +
        params[7] * T1X * T2V +
        params[8] * T2X * T2V
    )
    
    # Global scale for better BFGS conditioning
    scale = np.tanh(params[9])  # bounded scale in [-1, 1]
    
    return basis * scale
# EVOLVE-BLOCK-END
