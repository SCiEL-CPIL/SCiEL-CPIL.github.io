"""
Strategy: Universal Approximation via Dense Localized Gaussian Radial Basis Functions (RBFs).

This program abandons the global spectral/polynomial expansion approach (Chebyshev, sine, cosine, global polynomials)
in favor of a purely local interpolation model. By using a dense, evenly-spaced set of Gaussian RBFs across the domain
[-3, 3], the model can capture arbitrary local features, sharp transitions, and non-smooth behaviors that global
basis functions often smooth out or fail to resolve. The model remains linear in parameters, ensuring the evaluator's
BFGS optimizer converges robustly to the global least-squares solution without getting trapped in local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y given x and params using a dense RBF network.
    
    Args:
        x: Input array of shape (N,).
        params: Parameter array of shape (10,).
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x).ravel()
    
    # Define 10 evenly spaced centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Gaussian width parameter. 
    # Chosen to be narrow enough for local approximation but wide enough to avoid 
    # extreme ill-conditioning or dead zones between centers.
    # A width of 0.4 provides good overlap between adjacent centers (spacing 0.66).
    width = 0.4
    
    # Compute RBF values
    # d[i, j] = x[i] - centers[j]
    d = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF matrix: shape (N, 10)
    # phi(x, c) = exp(-((x-c)/width)^2)
    rbf_values = np.exp(-(d ** 2) / (width ** 2))
    
    # Linear combination: y = rbf_values @ params
    # This is a matrix-vector product, very efficient.
    y = rbf_values @ params
    
    return y

# EVOLVE-BLOCK-END
