import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Global linear trend + Local Gaussian RBFs.
    
    This basis is designed to be robust for BFGS optimization.
    - The first two parameters model a linear trend (1, x).
    - The next 8 parameters model local features using Gaussian RBFs
      with fixed centers and widths.
    
    This allows the model to capture both global trends and local
    variations (bumps, sharp changes) that global polynomials might
    struggle with or require high degrees to approximate.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
        
    # Basis 0: Constant
    # Basis 1: Linear
    # Basis 2-9: 8 Gaussian RBFs
    
    # Define centers and width for RBFs
    # Centers spread across the domain [-2, 2]
    # We use 8 centers.
    centers = np.array([-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4]) # Wait, 2.4 is outside [-2,2].
    # Let's use centers within [-2, 2].
    centers = np.linspace(-1.8, 1.8, 8)
    sigma = 0.5
    
    # Pre-compute RBF values
    # rbf_vals shape: (8, n)
    rbf_vals = np.zeros((8, n), dtype=np.float64)
    for i in range(8):
        rbf_vals[i, :] = np.exp(-((x - centers[i])**2) / (2 * sigma**2))
    
    # Construct full basis matrix (10, n)
    # Row 0: 1
    # Row 1: x
    # Rows 2-9: rbf_vals
    basis = np.zeros((10, n), dtype=np.float64)
    basis[0, :] = 1.0
    basis[1, :] = x
    basis[2:, :] = rbf_vals
    
    # Compute prediction
    result = basis.T @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
