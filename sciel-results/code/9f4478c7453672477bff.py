"""
Symbolic regression with a hybrid basis: polynomial + RBF + trigonometric terms.
The 10 parameters are weights for a carefully chosen set of basis functions.
This creates a linear-in-parameters model that BFGS can fit efficiently.
The basis includes:
- Low-order polynomials (constant, x, y, x^2, xy, y^2)
- RBF-like Gaussian bumps at key locations
- Trigonometric terms for periodic structures
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target value using a hybrid basis of polynomials, RBFs, and trig terms.
    
    Args:
        x: Input feature x (array of shape (n,))
        y: Input feature y (array of shape (n,))
        params: Weights for the 10 basis functions (array of shape (10,))
        
    Returns:
        Prediction array of shape (n,)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    
    # Ensure params is a flat array of at least length 10
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    weights = params[:10]
    
    # Build basis functions
    # 0: constant
    # 1: x
    # 2: y
    # 3: x^2
    # 4: xy
    # 5: y^2
    # 6: Gaussian at (0,0)
    # 7: Gaussian at (0.5, 0.5)
    # 8: sin(x) * cos(y)
    # 9: cos(x) * sin(y)
    
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Polynomial terms
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = y
    basis[:, 3] = x * x
    basis[:, 4] = x * y
    basis[:, 5] = y * y
    
    # RBF terms (Gaussian kernels)
    # Gaussian at origin
    dist_sq_0 = x * x + y * y
    basis[:, 6] = np.exp(-dist_sq_0 / (2.0 * 0.7 * 0.7))
    
    # Gaussian at (0.5, 0.5)
    dx = x - 0.5
    dy = y - 0.5
    dist_sq_1 = dx * dx + dy * dy
    basis[:, 7] = np.exp(-dist_sq_1 / (2.0 * 0.5 * 0.5))
    
    # Trigonometric interaction terms
    basis[:, 8] = np.sin(x) * np.cos(y)
    basis[:, 9] = np.cos(x) * np.sin(y)
    
    # Prediction is the dot product of basis (n, 10) and weights (10,)
    pred = basis @ weights
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
