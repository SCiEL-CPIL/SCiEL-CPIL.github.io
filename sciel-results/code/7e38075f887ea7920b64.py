"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using localized Gaussian RBFs plus a linear basis.
    Uses 8 Gaussian kernels centered at distinct points in [-3, 3] with varying widths,
    plus a constant term and a linear term.
    params[0] = constant coefficient
    params[1] = linear coefficient (multiplies x)
    params[2:10] = RBF amplitudes for 8 Gaussian kernels
    f(x) = params[0] + params[1]*x + sum_{k=2}^{9} params[k] * exp(-((x-c_k)^2)/(2*w_k^2))
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Define RBF centers and widths
    # Centers spread across [-3, 3] with some non-uniform spacing to capture features
    centers = np.array([-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.2, 2.8])
    widths = np.array([0.5, 0.7, 0.8, 1.0, 0.8, 0.7, 0.5, 0.4])
    
    # Build basis matrix: [1, x, RBF_0, ..., RBF_7]
    basis_matrix = np.empty((n, 10), dtype=np.float64)
    basis_matrix[:, 0] = 1.0  # constant
    basis_matrix[:, 1] = x    # linear
    
    for k in range(8):
        diff = x - centers[k]
        basis_matrix[:, k + 2] = np.exp(-(diff ** 2) / (2.0 * widths[k] ** 2))
    
    # Linear combination
    result = basis_matrix @ params
    
    return result
# EVOLVE-BLOCK-END
