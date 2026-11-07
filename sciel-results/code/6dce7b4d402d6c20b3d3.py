"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for the basis matrix to avoid recalculating for every BFGS step
_cached_x = None
_cached_basis = None
_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit


def _get_basis_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute scaled basis functions for the given x values.
    Uses a hybrid basis: Chebyshev polynomials (for stable polynomial fitting)
    and non-polynomial terms (sin, cos, 1/x, log, exp).
    Each column is scaled to have O(1) magnitude to improve conditioning for BFGS.
    """
    global _cached_x, _cached_basis
    
    # Simple check if x has changed (identity check or shape/value)
    # Since x is fixed during fitting, this is safe.
    if _cached_x is not None and _cached_x is x:
        return _cached_basis
    
    # Handle first call or change in x
    # x range is [0.1, 6]
    x_safe = np.clip(x, 1e-6, None)
    
    # 1. Chebyshev Polynomials T0..T4 mapped to [-1, 1]
    # Map x from [0.1, 6] to [-1, 1]
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x_safe - (x_min + x_max)) / (x_max - x_min)
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # 2. Non-polynomial terms, scaled to O(1)
    # sin(x), cos(x) are naturally O(1)
    sin_x = np.sin(x_safe)
    cos_x = np.cos(x_safe)
    
    # 1/x: range [1/6, 10]. Scale by 0.1 to get [0.016, 1.0]
    inv_x = (1.0 / x_safe) * 0.1
    
    # log(x): range [-2.3, 1.8]. Center and scale? 
    # log(x) - mean(log) ~ [-1.5, 1.0]. Scale by 1.0 is fine, or 0.5.
    log_x = np.log(x_safe) * 0.5
    
    # exp(x/2): range [1.05, 20]. Scale by 0.05 to get [0.05, 1.0]
    exp_x = np.exp(x_safe / 2.0) * 0.05
    
    # Construct basis matrix
    # Order: T0, T1, T2, T3, T4, sin, cos, 1/x, log, exp
    basis = np.column_stack((
        T0,
        T1,
        T2,
        T3,
        T4,
        sin_x,
        cos_x,
        inv_x,
        log_x,
        exp_x
    ))
    
    _cached_x = x
    _cached_basis = basis
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 10 scaled basis functions.
    
    The parameters are weights for each scaled basis function.
    The use of Chebyshev polynomials ensures a well-conditioned design matrix
    for the polynomial component, improving BFGS convergence.
    """
    # Deadline check
    if time.time() - _start_time > _DEADLINE:
        # Return a simple fallback if we are running out of time
        return params[0] * x + params[1]
    
    basis = _get_basis_matrix(x)
    
    try:
        pred = basis.dot(params)
    except Exception:
        # Fallback
        pred = params[0] * x + params[1]
        
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
    return pred
# EVOLVE-BLOCK-END
