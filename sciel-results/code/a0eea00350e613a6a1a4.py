"""Program for symbolic regression using a linear combination of basis functions."""
# EVOLVE-BLOCK-START
import numpy as np

# Cache for the basis matrix to avoid recalculating for every BFGS step
_cached_x = None
_cached_basis = None

def _get_basis_matrix(x):
    global _cached_x, _cached_basis
    # Simple check if x has changed (identity check or shape/value)
    # Since x is fixed during fitting, this is safe.
    if _cached_x is not None and _cached_x is x:
        return _cached_basis
    
    # Handle first call or change in x
    # RBF basis: 9 Gaussian RBFs + 1 bias term
    # Centers evenly spaced in [0.1, 6]
    # Width chosen to provide good coverage without excessive overlap
    
    x_safe = np.clip(x, 1e-6, None)
    
    # 9 RBF centers evenly spaced
    centers = np.linspace(0.1, 6.0, 9)
    # Width: ~0.8 gives good local coverage with some overlap
    width = 0.8
    
    # Compute RBF values: exp(-((x - center)^2) / (2 * width^2))
    # Shape: (n_points, 9)
    diff = x_safe[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / width) ** 2)
    
    # Add bias column
    bias = np.ones((x_safe.shape[0], 1))
    
    basis = np.hstack((rbf_values, bias))
    
    _cached_x = x
    _cached_basis = basis
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 9 RBF functions + 1 bias.
    params[0:9] are RBF weights, params[9] is bias.
    """
    basis = _get_basis_matrix(x)
    
    try:
        pred = basis.dot(params)
    except Exception:
        # Fallback: simple linear
        pred = params[0] * x + params[1]
        
    return pred
# EVOLVE-BLOCK-END
