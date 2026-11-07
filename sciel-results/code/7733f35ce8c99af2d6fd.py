# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear RBF Network with 9 fixed centers and 1 bias.
    
    Structure:
    - Input: x
    - Basis: 9 Gaussian Radial Basis Functions with fixed centers and width.
    - Output: Linear combination of basis functions + bias.
    
    Parameters:
    - params[0:9]: Weights for the 9 RBF centers
    - params[9]:   Bias term
    
    This model is linear in the parameters, making the optimization landscape
    convex (or at least much smoother) for BFGS compared to a non-linear MLP.
    The fixed centers ensure the model can capture local variations if the
    weights are optimized correctly.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
        
    # Define fixed RBF centers evenly spaced in [0.1, 6]
    # 9 centers
    n_centers = 9
    x_min, x_max = 0.1, 6.0
    centers = np.linspace(x_min, x_max, n_centers)
    
    # RBF width (sigma)
    # A width of 0.8 allows some overlap between neighbors, ensuring smoothness
    # while retaining local adaptability.
    sigma = 0.8
    
    # Calculate RBF activations
    # x: (N,), centers: (9,) -> x[:, None] - centers[None, :] is (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Linear combination
    # rbf_values: (N, 9), weights: (9,) -> (N,)
    y = rbf_values @ params[0:9] + params[9]
    
    return y
# EVOLVE-BLOCK-END
