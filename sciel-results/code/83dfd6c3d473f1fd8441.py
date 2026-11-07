"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich Chebyshev basis: T0 through T9.
    
    This formulation is linear in the parameters, making the optimization
    problem a convex quadratic (linear least squares). This ensures robust
    convergence for the BFGS optimizer, avoiding local minima associated
    with non-linear parameterization (like frequencies or rates).
    
    params[0:10] : coefficients for Chebyshev T0(z) through T9(z)
    """
    z = x / 3.0  # Map [-3, 3] to [-1, 1]
    
    # --- Chebyshev polynomials T0..T9 on z in [-1, 1] ---
    # Recurrence: T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
    
    T0 = np.ones_like(z)
    T1 = z
    
    T = [T0, T1]
    
    for n in range(2, 10):
        Tn = 2.0 * z * T[n-1] - T[n-2]
        T.append(Tn)
        
    # --- Assemble prediction ---
    pred = np.zeros_like(z)
    for i in range(10):
        pred += params[i] * T[i]
        
    return pred
# EVOLVE-BLOCK-END
