"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a cubic B-spline basis with 10 control points.
    
    The 10 parameters are the control point values of a cubic B-spline curve.
    This provides smooth, piecewise-polynomial approximation with high local flexibility.
    """
    start_time = time.time()
    deadline = 145.0

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # Normalize x to [0, 1] for B-spline knot placement
    t = (x + 3.0) / 6.0  # map [-3, 3] to [0, 1]

    # Cubic B-spline with 10 control points
    # Knot vector: open knot vector with 13 knots for cubic (degree 3)
    # For n_ctrl=10 control points, we need n_ctrl + degree + 1 = 14 knots
    # Open uniform knot vector: [0,0,0,0, 1/9, 2/9, ..., 8/9, 1,1,1,1]
    n_ctrl = 10
    degree = 3
    # Interior knots at i/(n_ctrl-1) for i=1,...,n_ctrl-2
    interior_knots = np.linspace(0, 1, n_ctrl + 1)[1:-1]
    # Full knot vector with degree+1 repeats at each end
    knots = np.concatenate([
        np.zeros(degree + 1),
        interior_knots,
        np.ones(degree + 1)
    ])
    # This gives 4 + (n_ctrl-2) + 4 = n_ctrl + 6 = 16 knots
    # For cubic B-spline with n_ctrl control points, we need n_ctrl + degree + 1 = 14 knots
    # Let me use the standard Cox-de Boor formulation

    # Use a simpler approach: compute cubic B-spline basis functions directly
    # For uniform cubic B-spline with 10 control points on [0,1]:
    # Control point spacing: h = 1/(n_ctrl - 1) = 1/9
    # Each control point i influences the interval [i*h - h, i*h + h] roughly

    h = 1.0 / (n_ctrl - 1)  # = 1/9

    # Precompute basis matrix using de Boor's algorithm / Cox-de Boor
    # For efficiency, we'll compute each of the 10 basis functions

    # The cubic B-spline basis function N_i,3(t) for control point i
    # with knots at positions corresponding to uniform spacing

    # Let's use the standard formulation:
    # With uniform knots at k_i = i * h for i = -3, -2, ..., n_ctrl
    # Control point i has knots k_i, k_{i+1}, k_{i+2}, k_{i+3}, k_{i+4}

    # Shift so that knot 0 is at position 0
    # Control point i corresponds to knots at i*h, (i+1)*h, ..., (i+4)*h
    # But we want the curve defined on [0, 1]

    # Simpler: use the cardinal cubic B-spline basis (uniform)
    # N_0(u) = (2 - 3u^2 + u^3)/2 for 0 <= u < 1
    # N_1(u) = (1 + 6u^2 - 3u^3)/2 for 0 <= u < 1
    # N_{-1}(u) = N_1(-u)
    # N_{-2}(u) = N_0(-u)

    # For non-uniform or clamped, let's just use a direct polynomial approach
    # Actually, let me use a different strategy: use 10 Lagrange-like basis functions
    # that form a partition of unity and are smooth

    # Alternative: use radial basis functions (Gaussian) centered at 10 equally spaced points
    # This is smooth, flexible, and easy to compute

    # Centers equally spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, n_ctrl)

    # Compute Gaussian RBF basis
    # Use a fixed width that balances locality and smoothness
    # Width ~ spacing between centers
    width = 6.0 / (n_ctrl - 1)  # = 2/3

    # For numerical stability, use a moderate width
    sigma = 0.8 * width  # slightly narrower than spacing

    # Compute all 10 basis values for all x points
    # diff[i, j] = x_i - center_j
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # Gaussian: exp(-diff^2 / (2 * sigma^2))
    basis_vals = np.exp(-diff**2 / (2.0 * sigma**2))

    # Stack: basis_vals has shape (n, 10)
    result = basis_vals @ params

    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)

    return result
# EVOLVE-BLOCK-END
