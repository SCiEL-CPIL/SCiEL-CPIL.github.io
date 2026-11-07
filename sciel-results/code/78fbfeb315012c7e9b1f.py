import numpy as np
import time

# Strategy: Hybrid Chebyshev + RBF Basis
#
# The current program (score 4384.74) uses a pure Chebyshev basis (T_0 to T_9).
# Inspiration 1 (score 4454.32) uses a hybrid basis: 6 Chebyshev terms + 4 RBFs.
# This hybrid approach is superior because:
# 1. Chebyshev polynomials capture global smooth trends efficiently with good conditioning.
# 2. RBFs (Radial Basis Functions) capture local features or deviations that a pure
#    polynomial basis of degree 5 might miss, without increasing the polynomial degree
#    (which would hurt conditioning).
# 3. The model remains linear in the 10 parameters, ensuring the external BFGS optimizer
#    faces a convex quadratic objective (MSE), which it can solve reliably.
#
# Design:
# - Map x from [-3, 3] to t in [-1, 1] via t = x/3.
# - Use Chebyshev polynomials T_0 to T_5 for the global component.
# - Use 4 Gaussian RBFs with fixed centers at [-2.25, -0.75, 0.75, 2.25] and fixed width
#   sigma=1.0 for the local component.
# - The 10 parameters are: 6 Chebyshev coefficients + 4 RBF amplitudes.
#
# This combines the structural robustness of the Chebyshev basis (from current/inspiration 1)
# with the local flexibility of RBFs (from inspiration 1), which was measured to improve
# the score from ~4385 to 4454.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Chebyshev + RBF Model.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:6] = Chebyshev coefficients for T_0(t) to T_5(t)
            params[6:10] = Amplitudes for 4 fixed RBFs
            
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # 1. Scale x to [-1, 1]
    # x is in [-3, 3], so t = x / 3.0
    t = x / 3.0
    
    # 2. Compute Chebyshev Polynomials T_0 to T_5
    # Using explicit formulas for speed and stability
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Combine Chebyshev part
    # p[0] * T0 + ... + p[5] * T5
    cheb_part = (p[0] * T0 + 
                 p[1] * T1 + 
                 p[2] * T2 + 
                 p[3] * T3 + 
                 p[4] * T4 + 
                 p[5] * T5)
                 
    # 3. RBF Part
    # 4 RBFs to utilize the remaining 4 params
    N_RBF = 4
    # Centers: Symmetrically placed in [-3, 3]
    # Avoiding the exact boundaries to ensure coverage
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    
    # Fixed width
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma**2
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (4,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    
    # rbf_values shape: (N, 4)
    # p[6:10] shape: (4,)
    rbf_part = rbf_values @ p[6:10]
    
    result = cheb_part + rbf_part
    
    return result
