"""
Gaussian Radial Basis Function (RBF) network with 10 parameters.
This replaces the linear Chebyshev basis with a fixed set of 10 Gaussian
RBFs centered at evenly spaced points in [0.1, 6]. The 10 parameters act
as output weights. This structure is linear in params, so BFGS converges
to the global optimum. RBFs provide local flexibility to fit arbitrary
smooth non-linear shapes without the conditioning issues of high-degree
polynomials or the global periodicity assumption of Fourier series.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF model with 10 parameters.
    params[0..9]: weights for each RBF center.
    
    1. Define 10 centers evenly spaced in [0.1, 6].
    2. Define a fixed width (sigma) for the Gaussians.
       A reasonable sigma is about 1/5 of the range, or range/10 for some overlap.
       Range = 5.9. Let's try sigma = 1.0 or 0.8.
       With 10 centers, spacing is ~0.59. Sigma ~ 0.5-0.8 gives good overlap.
       Let's use sigma = 0.6.
    3. Compute RBF values: exp(-((x - c_i)^2) / (2 * sigma^2))
    4. Prediction is the dot product of params and the RBF matrix.
    """
    x_min = 0.1
    x_max = 6.0
    n_params = 10
    
    # Centers evenly spaced
    centers = np.linspace(x_min, x_max, n_params)
    
    # Fixed width
    sigma = 0.6
    
    # Compute RBF values
    # x shape: (N, 1) or (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF values
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction
    prediction = rbf_values @ params
    
    return prediction
