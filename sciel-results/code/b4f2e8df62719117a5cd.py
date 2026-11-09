"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Build Vandermonde matrix for degree 5 polynomial in ascending powers
    # V[:, j] = x**j for j=0..5
    V = np.vander(x, N=6, increasing=True)
    
    # Add Tikhonov regularization to stabilize high-degree coefficients
    # Regularization matrix R: penalizes large coefficients, especially high-degree
    # Use a diagonal regularization that increases with degree to suppress oscillations
    R = np.diag([1.0, 1.0, 1.0, 10.0, 100.0, 1000.0])
    lam = 1e-4  # Regularization strength
    
    # Solve (V^T V + lam * R) c = V^T y
    A = V.T @ V + lam * R
    b = V.T @ y
    
    try:
        coeff = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        coeff = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Check time
    if time.time() - t_start > deadline:
        # Return best result found
        pass
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6, dtype=np.float64)
    
    return coeff
# EVOLVE-BLOCK-END
