"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Hybrid basis: 4 Gaussian RBFs + 6 Chebyshev polynomials
    # RBFs capture local/non-smooth features; Chebyshev captures global smooth trend
    
    # --- RBF Part: 4 Gaussian RBFs with fixed centers and widths ---
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    sigma = 0.7
    
    # Compute RBF terms: exp(-((x - c) / sigma)^2)
    # Shape: (N_points, 4)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-(diff / sigma) ** 2)
    
    # RBF contribution: sum(params[0:4] * rbf_terms)
    rbf_result = np.dot(rbf_terms, params[0:4])
    
    # --- Chebyshev Part: T_0 to T_5 ---
    # Scale x to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_5 via recurrence
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Chebyshev contribution: sum(params[4:10] * T_k)
    cheb_result = (
        params[4] * T0
        + params[5] * T1
        + params[6] * T2
        + params[7] * T3
        + params[8] * T4
        + params[9] * T5
    )
    
    result = rbf_result + cheb_result
    
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
