import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 piecewise linear 'hat' basis functions.
    
    This represents a structural change from global polynomial (Legendre) or 
    global trigonometric (Fourier) bases to a local basis. 
    Hat functions have compact support, allowing the model to capture 
    local features, sharp transitions, or localized bumps in the target 
    function, which global bases might smooth out or fail to represent 
    efficiently with only 10 coefficients.
    
    Basis:
      Centers c_i are evenly spaced in [-2, 2].
      Width w is the spacing between centers.
      phi_i(x) = max(0, 1 - |x - c_i| / w)
    """
    n_basis = 10
    # Define centers
    # Spacing
    w = 4.0 / (n_basis - 1)
    centers = np.linspace(-2.0, 2.0, n_basis)
    
    # Compute basis functions
    # x shape: (N,)
    # centers shape: (10,)
    # We want B shape: (N, 10)
    
    # |x[:, None] - centers[None, :]|
    dists = np.abs(x[:, None] - centers[None, :])
    
    # phi_i(x) = max(0, 1 - dists / w)
    B = np.maximum(0.0, 1.0 - dists / w)
    
    # Linear combination
    return B @ params
