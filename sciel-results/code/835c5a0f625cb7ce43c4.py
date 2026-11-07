"""Symbolic regression: recover hidden univariate f on [-3,3] with 10 free params.

STRATEGY: Use a Chebyshev polynomial basis of degree 9, which is orthogonal and
well-conditioned on [-1,1]. Since the model is strictly LINEAR in the 10 params,
BFGS converges to the exact least-squares solution in a handful of iterations.
This captures smooth functions far better than previous non-linear or mixed bases.

Model:
  f(x) = sum_{k=0}^{9} p_k * T_k(u),  where u = x/3 maps [-3,3] to [-1,1]

T_k are Chebyshev polynomials of the first kind:
  T_0(u) = 1
  T_1(u) = u
  T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)

This basis is optimal for approximating smooth functions on a bounded interval
with a fixed number of terms, and the linearity ensures BFGS finds the global
optimum without local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(x: np.ndarray, n: int = 10) -> np.ndarray:
    """Compute Chebyshev polynomial basis T_0..T_{n-1} evaluated at x.
    
    Args:
        x: input array, will be mapped to [-1, 1] via x/3
        n: number of basis functions (degree 0 to n-1)
    
    Returns:
        Array of shape (len(x), n) with basis values.
    """
    x = np.asarray(x, dtype=np.float64)
    u = x / 3.0  # map [-3, 3] to [-1, 1]
    
    basis = np.empty((x.size, n), dtype=np.float64)
    basis[:, 0] = 1.0
    if n > 1:
        basis[:, 1] = u
    for k in range(2, n):
        basis[:, k] = 2.0 * u * basis[:, k - 1] - basis[:, k - 2]
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    basis = _chebyshev_basis(x, n=10)
    return basis @ params
# EVOLVE-BLOCK-END
