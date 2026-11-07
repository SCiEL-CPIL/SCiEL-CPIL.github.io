"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10 localized Gaussian Radial Basis Functions (RBFs).
    
    The model is: f(x) = sum_{i=0}^{9} p[i] * exp(-((x - c_i)^2) / (2 * w_i^2))
    
    where c_i are fixed centers and w_i are fixed widths.
    The 10 parameters p[i] are the amplitudes of each RBF.
    
    Centers are spread across [-3, 3]:
      c_i = -3 + 6 * i / 9  for i = 0..9
    Widths vary to provide multi-scale approximation:
      w_i = 0.5 + 0.3 * (i % 3)  -> widths in {0.5, 0.8, 1.1}
    
    This is linear in parameters, so BFGS converges to the global least-squares
    minimum. The localized nature allows independent adjustment of local features.
    """
    # Deadline guard
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape[0] != 10:
        if p.size >= 10:
            p = p[:10]
        else:
            p = np.pad(p, (0, 10 - p.size))
    
    # Check deadline
    if time.time() - start_time > 145.0:
        return np.zeros(n, dtype=np.float64)
    
    # Fixed centers: 10 points evenly spaced in [-3, 3]
    centers = -3.0 + 6.0 * np.arange(10) / 9.0  # shape (10,)
    
    # Fixed widths: varying for multi-scale approximation
    # Pattern: 0.5, 0.8, 1.1, 0.5, 0.8, 1.1, 0.5, 0.8, 1.1, 0.5
    widths = np.array([0.5, 0.8, 1.1, 0.5, 0.8, 1.1, 0.5, 0.8, 1.1, 0.5])
    
    # Compute RBF values: shape (n, 10)
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # shape (n, 10)
    
    # Gaussian: exp(-diff^2 / (2 * w^2))
    # w2 shape (10,)
    w2 = 2.0 * widths ** 2  # shape (10,)
    
    rbf_values = np.exp(-(diff ** 2) / w2[np.newaxis, :])  # shape (n, 10)
    
    # Linear combination
    result = rbf_values @ p  # shape (n,)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
