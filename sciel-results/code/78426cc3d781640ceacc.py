"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise linear basis (hat functions) with 10 fixed knots evenly spaced
    across the domain [0.1, 6]. Linear-in-parameters for stable BFGS convergence.
    """
    # Domain
    x_min, x_max = 0.1, 6.0
    n_knots = 10
    
    # Fixed knots evenly spaced (including endpoints)
    knots = np.linspace(x_min, x_max, n_knots + 2)  # n_knots+2 points -> n_knots+1 intervals
    # Actually, for hat functions with n parameters, we need n knots.
    # Let's use n_knots = 10 interior knots + endpoints? 
    # Standard hat function basis: for N parameters, we can have N basis functions.
    # Let's define knots at evenly spaced points.
    
    # Simpler: 10 hat functions centered at 10 evenly spaced points.
    # But hat functions need support. Let's use a standard piecewise linear basis.
    
    # Alternative: Use 10 basis functions that are "tents" or "hats".
    # Let's place 10 knots evenly in [0.1, 6].
    knots = np.linspace(x_min, x_max, n_knots)
    
    # For each knot, create a hat function:
    # f_i(x) = max(0, 1 - |x - k_i| / h_i)
    # where h_i is the distance to the nearest neighbor knot (or half-distance to boundary).
    
    # To make it continuous and cover the domain, we can use:
    # For interior knots: h_i = min(k_{i+1} - k_i, k_i - k_{i-1})
    # For boundary knots: h_0 = k_1 - k_0, h_{n-1} = k_{n-1} - k_{n-2}
    
    n = len(knots)
    h = np.zeros(n)
    for i in range(n):
        if i == 0:
            h[i] = knots[1] - knots[0]
        elif i == n - 1:
            h[i] = knots[n-1] - knots[n-2]
        else:
            h[i] = min(knots[i+1] - knots[i], knots[i] - knots[i-1])
    
    # Build basis matrix: shape (len(x), n_knots)
    # f_i(x) = max(0, 1 - |x - knots[i]| / h[i])
    x_expanded = x[:, np.newaxis]  # (N, 1)
    knots_expanded = knots[np.newaxis, :]  # (1, n)
    
    dist = np.abs(x_expanded - knots_expanded)  # (N, n)
    basis = np.maximum(0, 1 - dist / h[np.newaxis, :])  # (N, n)
    
    # Prediction: basis @ params
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
