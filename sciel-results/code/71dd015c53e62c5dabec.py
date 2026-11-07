"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] != 10:
        p = p[:10] if p.size >= 10 else np.pad(p, (0, 10 - p.size))

    # Check deadline
    if time.time() > deadline:
        return np.zeros(n, dtype=np.float64)

    # RBF (Radial Basis Function) network
    # 10 Gaussian RBFs with fixed centers and widths
    # The 10 params are the RBF weights
    # This is a fundamentally different basis family from B-splines:
    # smooth, radially-symmetric, local basis functions
    
    # Fixed centers evenly spaced across [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width (sigma) for Gaussian RBFs
    # Use a width that gives good overlap: spacing is 6/9 = 0.667
    # sigma = spacing / 2 gives moderate overlap
    sigma = 0.5
    
    # Compute RBF basis: phi_i(x) = exp(-0.5 * ((x - c_i) / sigma)^2)
    # Shape: (10, n)
    diffs = x[None, :] - centers[:, None]  # (10, n)
    basis = np.exp(-0.5 * (diffs / sigma) ** 2)  # (10, n)
    
    # Linear combination
    result = basis @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
