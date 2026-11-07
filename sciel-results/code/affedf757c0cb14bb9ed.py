import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a basis of 10 Gaussian Radial Basis Functions.
    This is a materially different strategy from the current Chebyshev+RBF hybrid,
    moving to a purely local basis which can better capture non-smooth or highly
    varied features without the global constraints of polynomial bases.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Define 10 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define a width that ensures sufficient overlap between neighbors
    # Distance between centers is 6.0 / 9 = 0.666
    # Sigma = 0.8 ensures good overlap
    sigma = 0.8
    
    # Compute basis functions: phi_i(x) = exp(-0.5 * ((x - c_i) / sigma)^2)
    # Shape: (10, N)
    diff = x[None, :] - centers[:, None]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    # params shape (10,), basis shape (10, N)
    # result shape (N,)
    return basis.T @ params
