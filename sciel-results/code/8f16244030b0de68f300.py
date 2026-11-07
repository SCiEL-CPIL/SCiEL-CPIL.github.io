import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill

def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)

def _compute_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute a hybrid basis matrix of shape (n, 10).
    
    The basis combines:
    1. Low-degree Chebyshev polynomials T_0 to T_3 (4 terms) for global smooth trends.
    2. Six Gaussian RBFs with fixed centers and widths (6 terms) for local feature capture.
    
    This hybrid approach leverages the global approximation power of Chebyshev polynomials
    and the local flexibility of RBFs. The basis is linear in the parameters, ensuring
    BFGS converges to the global minimum of the MSE.
    
    Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight 1/sqrt(1-x^2),
    which helps with numerical conditioning. The RBFs add capacity to capture sharp
    features or non-polynomial behavior that a low-degree polynomial might miss.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    
    basis = np.zeros((n, 10), dtype=np.float64)
    
    # --- Part 1: Chebyshev Polynomials T_0 to T_3 ---
    # Scale x to [-1, 1] for standard Chebyshev definition
    u = x / 2.0
    
    # T_0 = 1
    basis[:, 0] = 1.0
    # T_1 = u
    basis[:, 1] = u
    # T_2 = 2*u^2 - 1
    basis[:, 2] = 2 * u * u - 1
    # T_3 = 4*u^3 - 3*u
    basis[:, 3] = 4 * u**3 - 3 * u
    
    # --- Part 2: Gaussian RBFs ---
    # Fixed centers spread across the domain [-2, 2]
    # Using 6 centers to cover the domain well
    centers = np.array([-1.5, -0.75, -0.25, 0.25, 0.75, 1.5])
    sigma = 0.5  # Fixed width
    
    for j in range(6):
        c = centers[j]
        # Gaussian: exp(-0.5 * ((x - c) / sigma)^2)
        basis[:, 4 + j] = np.exp(-0.5 * ((x - c) / sigma) ** 2)
        
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * basis_i(x).
    This is linear in params, so BFGS should converge to the global minimum.
    
    The hybrid Chebyshev + RBF basis provides both global and local approximation
    capabilities, which should improve performance over either basis alone.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        # Pad with zeros if needed
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
        
    # Check time budget
    if _remaining_time() < 1.0:
        # Return a simple fallback if time is running out
        return np.zeros_like(x)
    
    # Compute the basis matrix
    basis = _compute_basis(x)
    
    # Compute predictions: basis @ params
    pred = basis @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        # Replace non-finite values with 0
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        
    return pred
