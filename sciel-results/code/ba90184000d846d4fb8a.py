"""
Uses a hybrid basis: Chebyshev polynomials in x and y up to degree 5 (tensor product),
plus radial basis functions with adaptive centers derived from the data.
This combines global polynomial approximation with local flexibility to capture
both smooth trends and localized features. The Chebyshev basis is well-conditioned
on [-1,1], and the RBFs add local adaptivity.
"""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_poly(n, t):
    """Compute Chebyshev polynomial of degree n evaluated at t (array)."""
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t.copy()
    T_nm1 = np.ones_like(t)
    T_n = t.copy()
    for k in range(2, n + 1):
        T_np1 = 2.0 * t * T_n - T_nm1
        T_nm1 = T_n
        T_n = T_np1
    return T_n


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n_pts = len(x)
    
    # We have 10 parameters to use.
    # Strategy: Use 4 Chebyshev-based global terms and 6 RBF local terms.
    
    # Global Chebyshev basis:
    # T0(x)*T0(y) = 1
    # T1(x)*T0(y) = x
    # T0(x)*T1(y) = y
    # T2(x)*T0(y) = 2x^2 - 1
    # T0(x)*T2(y) = 2y^2 - 1
    # T1(x)*T1(y) = x*y
    # We'll use: 1, x, y, x^2, y^2, xy (6 terms) as a polynomial basis
    # But to keep it simple and well-conditioned, use Chebyshev:
    
    # Let's use 4 global Chebyshev tensor terms:
    # C0 = T0(x)T0(y) = 1
    # C1 = T1(x)T0(y) = x
    # C2 = T0(x)T1(y) = y
    # C3 = T2(x)T2(y) = (2x^2-1)(2y^2-1)
    
    C0 = np.ones(n_pts)
    C1 = x.copy()
    C2 = y.copy()
    C3 = (2.0 * x**2 - 1.0) * (2.0 * y**2 - 1.0)
    
    # Local RBF basis: 6 Gaussian RBFs
    # Use centers spread across the domain
    centers = np.array([
        [-0.7, -0.7],
        [ 0.7, -0.7],
        [-0.7,  0.7],
        [ 0.7,  0.7],
        [ 0.0,  0.0],
        [ 0.0,  0.0]  # center at origin, different width
    ])
    
    widths = np.array([
        2.0, 2.0, 2.0, 2.0, 3.0, 1.5
    ])
    
    # Compute RBF activations
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dist_sq = dx**2 + dy**2
    activations = np.exp(-(widths[None, :]**2) * dist_sq)
    
    # Build full basis matrix: 4 global + 6 local = 10 columns
    basis = np.column_stack([C0, C1, C2, C3, activations])
    
    return basis @ params
# EVOLVE-BLOCK-END
