"""Symbolic regression with Chebyshev polynomial basis on scaled interval."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on [0.1, 6].
    
    Maps x from [0.1, 6] to t in [-1, 1], then computes
    T_0(t), T_1(t), ..., T_9(t) as the 10 basis functions.
    
    Chebyshev polynomials are near-orthogonal on [-1, 1], providing
    excellent numerical conditioning for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to t in [-1, 1]
    # t = (2*x - (0.1 + 6)) / (6 - 0.1) = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Clip to [-1, 1] for numerical safety (shouldn't be needed for valid inputs)
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence:
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    basis = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * basis[n - 1] - basis[n - 2]
        basis.append(Tn)
    
    # Stack into design matrix (N, 10)
    X = np.column_stack(basis)
    
    # Prediction
    return X @ p
# EVOLVE-BLOCK-END
