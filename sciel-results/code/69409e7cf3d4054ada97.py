import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a hybrid basis designed for BFGS convergence:
    1-3: Low-order Chebyshev polynomials (global smoothness)
    4-10: Gaussian RBFs with varying widths (local flexibility)
    
    The RBF centers are evenly spaced, and widths vary to capture
    both broad and narrow features. This shifts the representation
    toward local interpolation while maintaining global smoothness.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Chebyshev Polynomials T_0 to T_2
    # T_0(t) = 1
    c0 = np.ones_like(t)
    # T_1(t) = t
    c1 = t
    # T_2(t) = 2t^2 - 1
    c2 = 2.0 * t**2 - 1.0
    
    # Gaussian RBFs with 7 centers and varying widths
    # Centers: evenly spaced from -3 to 3
    centers = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    
    # Varying widths: wider in the middle, narrower at edges
    # This allows capturing both broad trends and localized features
    sigmas = np.array([0.8, 1.0, 1.2, 1.5, 1.2, 1.0, 0.8])
    
    # Calculate Gaussians: exp(-(x - c)^2 / (2 * sigma^2))
    # x shape (N,), centers shape (7,)
    # diff shape (N, 7)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # sigma shape (7,) -> broadcast to (N, 7)
    sigma_sq = (sigmas[np.newaxis, :] ** 2)
    gaussians = np.exp(-(diff**2) / (2.0 * sigma_sq))
    
    # Build feature matrix (N, 10)
    # Columns 0-2: Chebyshev, Columns 3-9: Gaussians
    features = np.column_stack((c0, c1, c2, gaussians))
    
    # Linear combination
    result = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
