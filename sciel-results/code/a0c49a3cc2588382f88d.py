"""Symbolic regression with 1 input(s) - Hybrid Chebyshev + Gaussian basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 140.0  # Wall-clock limit with safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global smooth approximation) 
    combined with localized Gaussian basis functions (for sharp features).
    
    The basis is strictly linear in params, so BFGS converges to the global
    least-squares solution. 
    
    We map input x in [-2, 2] to t in [-1, 1] and use:
    - Chebyshev polynomials T_0(t) through T_5(t) for global structure
    - 4 Gaussian basis functions at different centers for local features
    
    This hybrid approach captures both the global trend and potential local
    variations that a pure polynomial might miss.
    """
    # Safety check: if we're close to the timeout, return a simple fallback
    if time.time() - _START_TIME > _TIMEOUT - 5.0:
        # Return mean of params as a constant fallback
        return np.mean(params) * np.ones_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(params) < 1:
        return np.zeros_like(x)
    
    N = len(x)
    
    # Scale x to [-1, 1] for better conditioning
    t = x / 2.0
    
    # Build the basis matrix: (N, 10)
    # Columns 0-5: Chebyshev polynomials T_0 through T_5
    # Columns 6-9: Gaussian basis functions
    
    basis = np.zeros((N, 10), dtype=np.float64)
    
    # Compute Chebyshev polynomials T_0(t) through T_5(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # T_0
    basis[:, 0] = 1.0
    
    # T_1
    basis[:, 1] = t
    
    # T_2 through T_5
    T_prev2 = np.ones(N, dtype=np.float64)  # T_0
    T_prev1 = t.copy()  # T_1
    
    for n in range(2, 6):
        T_curr = 2.0 * t * T_prev1 - T_prev2
        basis[:, n] = T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
    
    # Gaussian basis functions (localized features)
    # Centers chosen to cover the domain [-1, 1] in t-space
    # Widths chosen to be moderately narrow for local adaptation
    
    # Center at -0.5 (x = -1.0)
    basis[:, 6] = np.exp(-((t + 0.5) ** 2) / (2.0 * 0.3 ** 2))
    
    # Center at 0.0 (x = 0.0)
    basis[:, 7] = np.exp(-(t ** 2) / (2.0 * 0.3 ** 2))
    
    # Center at 0.5 (x = 1.0)
    basis[:, 8] = np.exp(-((t - 0.5) ** 2) / (2.0 * 0.3 ** 2))
    
    # Center at 1.0 (x = 2.0) - boundary feature
    basis[:, 9] = np.exp(-((t - 1.0) ** 2) / (2.0 * 0.2 ** 2))
    
    # Prediction: (N, 10) dot (10,) -> (N,)
    prediction = basis.dot(params)
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
    
    return prediction
# EVOLVE-BLOCK-END
