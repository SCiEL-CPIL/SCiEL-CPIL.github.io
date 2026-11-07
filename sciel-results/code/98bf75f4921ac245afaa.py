"""Program for symbolic regression using a linear combination of standardized basis functions."""
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
    # Basis functions chosen for coverage and conditioning over [0.1, 6]
    # 1, x, x^2, x^3, x^4, x^5, exp(x/2), exp(-x/2), log(x+1), sin(x)
    
    x_safe = np.clip(x, 1e-6, None)
    
    n = len(x_safe)
    basis = np.zeros((n, 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    basis[:, 1] = x_safe
    basis[:, 2] = x_safe ** 2
    basis[:, 3] = x_safe ** 3
    basis[:, 4] = x_safe ** 4
    basis[:, 5] = x_safe ** 5
    basis[:, 6] = np.exp(x_safe / 2.0)
    basis[:, 7] = np.exp(-x_safe / 2.0)
    basis[:, 8] = np.log(x_safe + 1.0)
    basis[:, 9] = np.sin(x_safe)
    
    # Normalize each column by its L2 norm to improve conditioning
    # This makes the Hessian of the least-squares problem better conditioned
    for j in range(10):
        col = basis[:, j]
        norm = np.linalg.norm(col)
        if norm > 1e-12:
            basis[:, j] = col / norm
        else:
            basis[:, j] = col
    
    _cached_x = x
    _cached_basis = basis
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 10 L2-normalized basis functions.
    """
    basis = _get_basis_matrix(x)
    
    try:
        pred = basis.dot(params)
    except Exception:
        # Fallback to simple linear model if something goes wrong
        pred = params[0] * x + params[1]
        
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
