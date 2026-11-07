"""Symbolic regression with a well-conditioned mixed Chebyshev + periodic basis."""
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using 10 parameters.
    
    Basis design:
    - x is in [0.1, 6]. We map to t = (x - 3.05) / 2.95 -> [-1, 1]
    - We use a mix of Chebyshev polynomials and periodic terms.
      6 Chebyshev polynomials T_0 through T_5 for smooth polynomial behavior,
      and 4 periodic terms (sin/cos at different frequencies) to capture
      any oscillatory components in the target function.
    
    Basis functions (10 total):
    b0 = T_0(t) = 1
    b1 = T_1(t) = t
    b2 = T_2(t) = 2t^2 - 1
    b3 = T_3(t) = 4t^3 - 3t
    b4 = T_4(t) = 8t^4 - 8t^2 + 1
    b5 = T_5(t) = 16t^5 - 20t^3 + 5t
    b6 = sin(pi * t)
    b7 = cos(pi * t)
    b8 = sin(2*pi * t)
    b9 = cos(2*pi * t)
    
    This gives BFGS a smooth, well-conditioned landscape for a broader
    class of target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1]
    t = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 through T_5 using recurrence
    # T_0(t) = 1, T_1(t) = t, T_{n+1}(t) = 2t*T_n(t) - T_{n-1}(t)
    T = np.zeros((6, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 6):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Periodic terms
    sin_pi_t = np.sin(np.pi * t)
    cos_pi_t = np.cos(np.pi * t)
    sin_2pi_t = np.sin(2.0 * np.pi * t)
    cos_2pi_t = np.cos(2.0 * np.pi * t)
    
    # Build full basis matrix (10 x N)
    basis = np.zeros((10, len(x)), dtype=np.float64)
    basis[:6, :] = T
    basis[6, :] = sin_pi_t
    basis[7, :] = cos_pi_t
    basis[8, :] = sin_2pi_t
    basis[9, :] = cos_2pi_t
    
    # Linear combination: pred = sum(params[i] * basis[i])
    pred = np.sum(params[:, np.newaxis] * basis, axis=0)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
