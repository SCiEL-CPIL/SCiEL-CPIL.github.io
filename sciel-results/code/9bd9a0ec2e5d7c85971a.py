"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a Gaussian RBF basis.
    
    The model is linear in the 10 parameters: y = sum(params[i] * RBF_i(x)).
    The RBF centers are evenly spaced in [0.1, 6] with a fixed width.
    This provides a flexible, well-conditioned basis for smooth functions.
    
    Key design choices:
    - 10 Gaussian RBFs with centers at linspace(0.1, 6, 10)
    - Fixed width sigma = 0.8 (chosen to balance locality and coverage)
    - No normalization of basis (RBFs are already bounded in [0,1])
    - This avoids the conditioning issues of polynomial bases
    """
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 100

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 1.0)
        
    n = x.shape[0]
    if n == 0:
        return np.array([])

    # RBF centers: evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width
    sigma = 0.8
    
    # Compute RBF basis: B[i, j] = exp(-0.5 * ((x[i] - centers[j]) / sigma)^2)
    # Shape: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    B = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Check for non-finite values in basis
    if not np.all(np.isfinite(B)):
        B = np.where(np.isfinite(B), B, 0.0)
        
    # Prediction: linear combination
    prediction = B @ p
    
    # Check for non-finite values in prediction
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
