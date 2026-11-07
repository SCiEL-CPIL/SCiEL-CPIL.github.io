import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid Fourier (trigonometric) basis.
    
    The model maps x in [-2, 2] to t in [-pi, pi] and uses the first 10 terms
    of the Fourier series: {1, cos(t), sin(t), cos(2t), sin(2t), ..., cos(5t), sin(5t)}.
    
    This provides an orthogonal, periodic representation that is well-conditioned
    for BFGS optimization and can capture smooth, oscillatory, or bounded global
    structures.
    
    y(x) = a_0 + sum_{k=1}^{5} [a_k * cos(k*t) + b_k * sin(k*t)]
    where t = pi * x / 2
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure x is 1D for consistent broadcasting
    x_flat = x.flatten()
    
    # Map x in [-2, 2] to t in [-pi, pi]
    t = np.pi * x_flat / 2.0
    
    # Build the Fourier basis matrix: shape (N_points, 10)
    # Basis functions: 1, cos(t), sin(t), cos(2t), sin(2t), ..., cos(5t), sin(5t)
    N = len(x_flat)
    basis = np.zeros((N, 10), dtype=np.float64)
    
    # Constant term
    basis[:, 0] = 1.0
    
    # Cosine and sine terms for k = 1 to 5
    idx = 1
    for k in range(1, 6):
        kt = k * t
        basis[:, idx] = np.cos(kt)
        basis[:, idx + 1] = np.sin(kt)
        idx += 2
    
    # Compute prediction: sum of weights * basis
    # params shape (10,), basis shape (N, 10)
    # result shape (N,)
    result = basis @ params
    
    # Reshape back to original x shape
    result = result.reshape(x.shape)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
