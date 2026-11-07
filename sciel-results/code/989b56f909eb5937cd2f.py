"""
Symbolic regression using Chebyshev polynomial basis.
The hidden function is approximated by a linear combination of Chebyshev polynomials
of the first kind, scaled to the domain [-1, 1].
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    
    Uses a hybrid Chebyshev + RBF basis:
    - First 6 params: Chebyshev polynomials T_0..T_5
    - Last 4 params: Gaussian RBFs centered at -1.5, -0.5, 0.5, 1.5
    """
    # Deadline guard: track start time
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    n = len(x)
    n_params = 10
    
    # --- Part 1: Chebyshev polynomials (first 6) ---
    # Scale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    n_cheb = 6
    basis = np.zeros((n, n_params))
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    # T_1(t) = t
    basis[:, 1] = t
    
    for k in range(2, n_cheb):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # --- Part 2: Gaussian RBFs (last 4) ---
    # Centers spread across [-2, 2]
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    width = 1.0  # Gaussian width
    
    for i, c in enumerate(centers):
        idx = n_cheb + i
        # Gaussian RBF: exp(-((x - c)^2) / (2 * width^2))
        diff = x - c
        basis[:, idx] = np.exp(-0.5 * (diff / width) ** 2)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > DEADLINE:
        # Return current best estimate (linear combination so far)
        return basis @ params
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
