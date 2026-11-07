import numpy as np
from scipy.special import factorial
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 Legendre polynomials scaled to [-2, 2].
    
    Legendre polynomials P_n(t) are orthogonal on [-1, 1] with respect to the
    uniform weight. Since data is drawn uniformly from [-2, 2], the scaled
    basis P_n(x/2) is orthogonal w.r.t. the empirical distribution.
    
    This gives a well-conditioned design matrix, enabling BFGS to converge
    robustly to the global minimum of the convex MSE loss.
    
    The basis spans smooth functions up to degree 9, which can capture
    polynomials, sinusoidal-like behavior, and other smooth nonlinear targets.
    """
    n_basis = 10
    t = x / 2.0  # scale to [-1, 1]
    
    # Compute Legendre polynomials P_0 through P_9 using recurrence
    P = np.zeros((len(x), n_basis))
    P[:, 0] = 1.0
    if n_basis > 1:
        P[:, 1] = t
        for n in range(1, n_basis - 1):
            P[:, n + 1] = ((2 * n + 1) * t * P[:, n] - n * P[:, n - 1]) / (n + 1)
    
    return P @ params
