import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Low-degree polynomials and Gaussian Radial Basis Functions.
    This approach captures global trends via polynomials and local features via RBFs.
    """
    # 1. Polynomial terms: 1, x, x^2, x^3
    p0 = np.ones_like(x)
    p1 = x
    p2 = x ** 2
    p3 = x ** 3
    
    # 2. Gaussian RBF terms
    # Centers chosen to cover [-3, 3]
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    sigma = 1.0
    
    # Compute Gaussians: shape (6, N)
    # x shape (N,), centers shape (6,)
    # (x[:, None] - centers[None, :]) ** 2
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Stack all basis functions
    # p0..p3 are shape (N,)
    # gaussians is shape (6, N)
    basis = np.vstack([
        p0[np.newaxis, :],
        p1[np.newaxis, :],
        p2[np.newaxis, :],
        p3[np.newaxis, :],
        gaussians
    ])
    
    # params has length 10
    # coeffs = params[:10]
    # Result: (10,) @ (10, N) -> (N,)
    return params @ basis
