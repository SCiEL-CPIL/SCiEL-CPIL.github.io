"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute a linear combination of 10 localized Gaussian Radial Basis Functions (RBFs)
    centered at distinct points in [-3, 3] with varying widths.
    
    This localized basis allows the 10 parameters to independently adjust local amplitude
    to approximate arbitrary non-linear features that global polynomials or splines may underfit.
    The basis is linear in params, allowing scipy.optimize.minimize (BFGS) to fit the coefficients.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n == 0:
        return np.array([])

    # Define centers and widths for 10 RBFs
    # Centers spread across [-3, 3]
    centers = np.array([-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
    
    # Varying widths to capture different scales of variation
    # Smaller widths for sharper features, larger for broader trends
    # We use a mix of narrow and wide RBFs
    widths = np.array([0.5, 0.8, 1.2, 1.5, 2.0, 2.0, 1.5, 1.2, 0.8, 0.5])

    # Compute RBF basis matrix
    # basis[i, j] = exp(-0.5 * ((x_j - centers[i]) / widths[i])^2)
    # Vectorized computation
    # diff shape: (10, n)
    diff = (x[None, :] - centers[:, None]) / widths[:, None]
    basis = np.exp(-0.5 * diff**2)

    # result = basis.T @ params
    # basis is (10, n), params is (10,) -> result is (n,)
    result = basis.T @ params

    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)

    # Check deadline (computation above is very fast, but guard just in case)
    if time.time() > deadline:
        pass

    return result
# EVOLVE-BLOCK-END
