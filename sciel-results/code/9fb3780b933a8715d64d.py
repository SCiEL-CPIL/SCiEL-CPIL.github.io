"""Program for symbolic regression using a Gaussian Radial Basis Function (RBF) network."""
# EVOLVE-BLOCK-START
import numpy as np

# Define fixed centers and width for the RBFs
# Domain is [0.1, 6.0]
# We use 10 RBFs, so 10 parameters will be the weights for each.
# Evenly spaced centers
_START = 0.1
_END = 6.0
_N_RBF = 10
_CENTERS = np.linspace(_START, _END, _N_RBF)
# Width sigma: chosen to ensure significant overlap between neighbors
# Spacing is (6.0 - 0.1) / 9 = 0.655...
# Sigma of 0.6 provides good local coverage
_SIGMA = 0.6
_SIGMA_SQ_2 = 2.0 * _SIGMA**2

# Pre-compute the squared distances? No, x changes during evaluation calls?
# Actually, x is fixed during the fitting process (BFGS calls equation with same x, different params).
# But equation might be called with different x for testing.
# We can cache the RBF matrix for a specific x array to speed up BFGS iterations.

_cached_x = None
_cached_rbf_matrix = None

def _get_rbf_matrix(x):
    global _cached_x, _cached_rbf_matrix
    
    # Check if x is the same object or has the same values
    # Using identity check first for speed, then fallback to value check if needed
    # Since x is a numpy array passed by the evaluator, it might be the same object during fit.
    if _cached_x is not None:
        if _cached_x is x:
            return _cached_rbf_matrix
        # If not same object, check if values are identical (expensive, but safe)
        # For now, we assume if it's not the same object, it's a new evaluation context
        # Or we can just recompute. Recalculating 10 Gaussians for 200 points is cheap.
        
    # Compute RBF values
    # x shape: (N,)
    # Centers shape: (10,)
    # We want matrix of shape (N, 10)
    
    # (x[:, None] - centers[None, :])
    diff = x[:, None] - _CENTERS[None, :]
    sq_diff = diff * diff
    
    # exp(-sq_diff / (2 * sigma^2))
    rbf_vals = np.exp(-sq_diff / _SIGMA_SQ_2)
    
    _cached_x = x
    _cached_rbf_matrix = rbf_vals
    
    return rbf_vals

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 10 Gaussian RBFs.
    params: array of length 10, weights for each RBF.
    """
    # Ensure x is 1D
    x = np.asarray(x).flatten()
    
    # Get the RBF matrix
    # If x is very small, flatten might change shape, but x is 1D input.
    rbf_matrix = _get_rbf_matrix(x)
    
    # Linear combination
    # rbf_matrix: (N, 10)
    # params: (10,)
    pred = rbf_matrix @ params
    
    return pred
# EVOLVE-BLOCK-END
