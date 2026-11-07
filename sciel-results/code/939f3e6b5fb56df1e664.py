"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 fixed Chebyshev polynomials (global smooth) + 5 learnable RBF kernels (local detail).
    
    First 5 params: Chebyshev coefficients (T_0 to T_4)
    Last 5 params: RBF amplitudes
    RBF centers are fixed at evenly spaced points on [-2, 2] to ensure stability.
    
    This combines the well-conditioned global approximation of Chebyshev with local flexibility.
    """
    # Map x from [-2, 2] to [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    T0 = np.ones_like(x)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Global part: linear combination of Chebyshev polynomials
    cheb_coeffs = params[:5]
    global_part = cheb_coeffs[0] * T0 + cheb_coeffs[1] * T1 + cheb_coeffs[2] * T2 + cheb_coeffs[3] * T3 + cheb_coeffs[4] * T4
    
    # Local part: RBF with fixed centers and learnable amplitudes
    # Fixed centers evenly spaced: -2, -1, 0, 1, 2
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    amplitudes = params[5:]
    
    # Fixed width for Gaussian kernels
    width = 0.5
    inv_2w2 = 1.0 / (2.0 * width * width)
    
    # Compute Gaussian RBFs
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponents = -diff * diff * inv_2w2
    basis = np.exp(exponents)  # shape: (n_points, 5)
    
    # Linear combination with amplitudes
    local_part = basis @ amplitudes
    
    # Combine global and local parts
    result = global_part + local_part
    
    return result
# EVOLVE-BLOCK-END
