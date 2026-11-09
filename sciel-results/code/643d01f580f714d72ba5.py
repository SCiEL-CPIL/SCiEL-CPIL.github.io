"""Piecewise linear hat-basis regression with fixed knots for global linear least-squares fit."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global piecewise linear 'hat' basis with fixed evenly spaced knots over [-2, 2].
    
    We place K interior knots plus 2 boundary knots, giving K+2 basis functions total.
    Since we have 10 params, we use K = 8 interior knots -> 10 basis functions.
    
    The basis functions are the standard linear B-splines (hat functions) on a uniform grid.
    The model is strictly linear in params, so BFGS (or any least-squares solver) finds
    the global optimum.
    """
    # Number of basis functions = len(params) = 10
    n_basis = len(params)
    
    # Fixed knots: evenly spaced across [-2, 2]
    # We need n_basis basis functions. For linear B-splines with n_basis functions,
    # we have n_basis + 1 knots (including boundaries).
    # Actually, for linear B-splines: number of basis functions = number of intervals + 1
    # Let's define knots explicitly.
    
    # With n_basis = 10, we want 10 linear hat functions.
    # Standard approach: define n_basis + 1 knots, but the first and last are boundary.
    # Let's use: knots = np.linspace(-2, 2, n_basis + 1)
    # Then basis function i is the hat centered at knot i, for i in 1..n_basis-1
    # and the first and last are half-hats at the boundaries.
    
    knots = np.linspace(-2.0, 2.0, n_basis + 1)
    
    # Compute basis function values for each x
    # For each x, find which interval it falls in
    # basis[i](x) is the linear hat function associated with knot[i]
    
    # Efficient computation:
    # For each x, determine the interval index
    # x is in [knots[j], knots[j+1]] for some j
    
    # Clip x to domain for safety
    x_clipped = np.clip(x, knots[0], knots[-1])
    
    # Find interval index: j such that knots[j] <= x < knots[j+1]
    # Use searchsorted
    interval_idx = np.searchsorted(knots, x_clipped, side='right') - 1
    interval_idx = np.clip(interval_idx, 0, len(knots) - 2)
    
    # Local coordinate within interval
    x0 = knots[interval_idx]
    x1 = knots[interval_idx + 1]
    dx = x1 - x0
    t = (x_clipped - x0) / dx  # in [0, 1)
    
    # Basis functions:
    # For interior knots (i = 1, ..., n_basis-1), hat function is nonzero on 
    # [knots[i-1], knots[i+1]]
    # For boundary knots (i=0 and i=n_basis-1), half-hat functions
    
    # Build basis matrix: shape (len(x), n_basis)
    basis = np.zeros((len(x), n_basis))
    
    # For each point, it contributes to at most 2 basis functions:
    # the left knot (interval_idx) and the right knot (interval_idx + 1)
    
    # Left knot contribution: 1 - t
    basis[np.arange(len(x)), interval_idx] = 1.0 - t
    
    # Right knot contribution: t
    basis[np.arange(len(x)), interval_idx + 1] = t
    
    # Prediction: linear combination of basis functions with params
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
