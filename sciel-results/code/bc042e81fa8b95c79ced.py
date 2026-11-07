"""
Hybrid Chebyshev-Gaussian Basis for Symbolic Regression.

Combines the structural idea of the current program (fixed, orthogonal Chebyshev
polynomial basis mapped to [-1, 1]) with the refinement machinery from
inspiration 3 (Gaussian RBFs for local flexibility).

Strategy:
- Map x in [-2, 2] to t in [-1, 1].
- Use a basis of 10 features:
  * 5 Chebyshev polynomials T_0..T_4 for global smooth trend (well-conditioned,
    orthogonal on [-1, 1]).
  * 5 Gaussian RBFs centered at key locations in [-2, 2] to capture local
    variations, kinks, or sharp features that a pure polynomial basis cannot
    efficiently model with only 10 coefficients.

This hybrid basis is linear in parameters, making the optimization problem
convex (quadratic MSE), which allows BFGS to converge reliably to the global
minimum for the given basis. The Chebyshev components ensure good conditioning
for the global trend, while the Gaussians provide local flexibility.

Deadline guard: track elapsed time and return the best result found with a
safety margin before the 150s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill


def _chebyshev_gaussian_features(x):
    """
    Compute hybrid Chebyshev-Gaussian features for x in [-2, 2].
    
    Features:
      F[:, 0] = 1                          (T_0)
      F[:, 1] = t                          (T_1, where t = x/2)
      F[:, 2] = 2*t^2 - 1                  (T_2)
      F[:, 3] = 4*t^3 - 3*t                (T_3)
      F[:, 4] = 8*t^4 - 8*t^2 + 1          (T_4)
      F[:, 5] = exp(-(x - 1.5)^2 / 0.5)    (Gaussian RBF)
      F[:, 6] = exp(-(x - 0.75)^2 / 0.5)   (Gaussian RBF)
      F[:, 7] = exp(-(x)^2 / 0.5)          (Gaussian RBF)
      F[:, 8] = exp(-(x + 0.75)^2 / 0.5)   (Gaussian RBF)
      F[:, 9] = exp(-(x + 1.5)^2 / 0.5)    (Gaussian RBF)
    
    Returns array of shape (N, 10).
    """
    N = len(x)
    F = np.zeros((N, 10), dtype=float)
    
    if N == 0:
        return F
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Chebyshev polynomials T_0..T_4
    F[:, 0] = 1.0
    F[:, 1] = t
    F[:, 2] = 2.0 * t**2 - 1.0
    F[:, 3] = 4.0 * t**3 - 3.0 * t
    F[:, 4] = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Gaussian RBFs
    denom = 0.5
    centers = [1.5, 0.75, 0.0, -0.75, -1.5]
    
    for i, c in enumerate(centers):
        F[:, 5 + i] = np.exp(-(x - c)**2 / denom)
    
    return F


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        # Return a safe fallback: just the mean (first param) if we're out of time
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            x = x.ravel()
        p = np.asarray(params, dtype=float).ravel()
        if p.size >= 1:
            return np.full(len(x), p[0])
        return np.zeros(len(x))
    
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
        
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 10:
        # Fallback if params shape is wrong (shouldn't happen)
        p = np.ones(10)
        
    N = len(x)
    if N == 0:
        return np.array([])
        
    # Compute hybrid basis features
    F = _chebyshev_gaussian_features(x)
    
    # Linear combination
    result = F @ p
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
