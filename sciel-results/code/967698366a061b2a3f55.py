"""Symbolic regression with Chebyshev basis and tensor-product structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Chebyshev polynomials up to degree 5
# T_0 = 1, T_1 = x, T_2 = 2x^2 - 1, T_3 = 4x^3 - 3x, T_4 = 8x^4 - 8x^2 + 1, T_5 = 16x^5 - 20x^3 + 5x

def _chebyshev(x, max_deg=5):
    """Compute Chebyshev polynomials T_0(x) through T_max_deg(x) for array x."""
    n = len(x)
    T = np.zeros((n, max_deg + 1))
    T[:, 0] = 1.0
    if max_deg >= 1:
        T[:, 1] = x
    for k in range(2, max_deg + 1):
        T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
    return T


def _build_basis(x, y, max_deg=3):
    """
    Build a tensor-product Chebyshev basis.
    For degrees (i, j) with i+j <= max_deg, include T_i(x) * T_j(y).
    
    With max_deg=3, we have:
    (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    That's 10 basis functions, matching the 10 params.
    """
    Tx = _chebyshev(x, max_deg)
    Ty = _chebyshev(y, max_deg)
    
    # Collect basis functions for all (i,j) with i+j <= max_deg
    basis_funcs = []
    for i in range(max_deg + 1):
        for j in range(max_deg + 1 - i):
            basis_funcs.append((i, j))
    
    # There should be exactly 10 for max_deg=3
    n_basis = len(basis_funcs)
    n_points = len(x)
    B = np.zeros((n_points, n_basis))
    
    for idx, (i, j) in enumerate(basis_funcs):
        B[:, idx] = Tx[:, i] * Ty[:, j]
    
    return B, basis_funcs


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using Chebyshev tensor-product basis.
    params: array of length 10 (coefficients for each basis function)
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    max_deg = 3
    B, _ = _build_basis(x, y, max_deg)
    
    # Prediction is B @ params
    pred = B @ params
    
    return pred
# EVOLVE-BLOCK-END
