"""
Strategy: Radial Basis Function (RBF) approximation.
Instead of a global polynomial, we use a set of fixed Gaussian basis functions centered at specific points in the input domain.
The parameters scale the amplitude of each basis function.
This allows the model to capture local variations and non-polynomial shapes more effectively than a global quadratic.
We use 9 Gaussian centers + 1 bias term = 10 parameters.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has length 10
    if len(p) < 10:
        # Pad with zeros if necessary, though evaluator should provide 10
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]
        
    # Define fixed centers in the domain [0.5, 2.0]
    # We choose a 2x2x2 grid of centers to cover the space, plus one center at the midpoint for a global-ish component.
    # Centers:
    # c0: (0.5, 0.5, 0.5)
    # c1: (2.0, 0.5, 0.5)
    # c2: (0.5, 2.0, 0.5)
    # c3: (2.0, 2.0, 0.5)
    # c4: (0.5, 0.5, 2.0)
    # c5: (2.0, 0.5, 2.0)
    # c6: (0.5, 2.0, 2.0)
    # c7: (2.0, 2.0, 2.0)
    # c8: (1.25, 1.25, 1.25) - Center of the domain
    
    # To make the basis functions well-conditioned for BFGS starting at 1.0,
    # we want them to have significant overlap but distinct peaks.
    # Domain size is 1.5. A width sigma of 0.5 gives a range of influence of ~1.5 (3 sigma).
    # This means each point is influenced by multiple centers, creating a smooth interpolation.
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25]
    ])
    
    # 9 centers + 1 bias = 10 params
    # params[0..8] are amplitudes for the 9 Gaussians
    # params[9] is the bias (constant offset)
    
    # Compute distances from each input point to each center
    # x, y, z are arrays of shape (N,)
    # We need to compute ||[x_i, y_i, z_i] - c_j||^2 for all i, j
    
    # Reshape for broadcasting: (N, 1, 3) - (1, 9, 3) -> (N, 9, 3)
    inputs = np.stack([x, y, z], axis=-1) # Shape (N, 3)
    inputs_expanded = inputs[:, np.newaxis, :] # Shape (N, 1, 3)
    centers_expanded = centers[np.newaxis, :, :] # Shape (1, 9, 3)
    
    # Squared distances: (N, 9)
    diff = inputs_expanded - centers_expanded
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Gaussian kernel: exp(-dist_sq / (2 * sigma^2))
    # Let's choose sigma = 0.6. 
    # 2 * sigma^2 = 0.72
    # If dist_sq is small (near center), exp(0) = 1.
    # If dist_sq is large (far away), exp(-large) ~ 0.
    # This ensures local control.
    
    sigma_sq = 0.6**2
    denom = 2.0 * sigma_sq
    
    # Compute basis values: (N, 9)
    basis = np.exp(-dist_sq / denom)
    
    # Linear combination: (N, 9) * (9,) -> (N,)
    # Add bias
    pred = basis @ p[:9] + p[9]
    
    return pred
