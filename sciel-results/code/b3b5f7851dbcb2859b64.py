import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 RBF (Gaussian) basis functions.
    
    This combines the structural idea of local basis functions (from the hat function
    inspiration) with a smoother, differentiable representation. Unlike hat functions
    which are piecewise linear and have discontinuous first derivatives, Gaussian RBFs
    are infinitely differentiable and provide smooth local approximations.
    
    The centers are evenly spaced in [-2, 2]. The width (sigma) is set to the spacing
    between centers, providing a natural scale. This allows the model to capture
    localized features, bumps, or sharp transitions that global polynomial or Fourier
    bases may struggle with when limited to only 10 coefficients.
    
    Basis:
      phi_i(x) = exp(-0.5 * ((x - c_i) / sigma)^2)
      where c_i are evenly spaced centers in [-2, 2]
      and sigma is the spacing between centers.
    """
    n_basis = 10
    
    # Define centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, n_basis)
    
    # Width (sigma) is the spacing between centers
    sigma = 4.0 / (n_basis - 1)
    
    # Compute basis functions
    # x shape: (N,)
    # centers shape: (10,)
    # We want B shape: (N, 10)
    
    # (x[:, None] - centers[None, :]) / sigma
    scaled_dists = (x[:, None] - centers[None, :]) / sigma
    
    # phi_i(x) = exp(-0.5 * scaled_dists^2)
    B = np.exp(-0.5 * scaled_dists ** 2)
    
    # Linear combination
    return B @ params
