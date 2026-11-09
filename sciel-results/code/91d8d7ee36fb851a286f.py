import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed, well-conditioned linear basis using logistic (sigmoid) functions.
    
    This is a structural departure from polynomial bases. By using a fixed linear model 
    y = sum p_i * sigmoid(a_i * (x - c_i)), the optimization problem remains convex 
    (linear least squares in the parameters). BFGS will converge rapidly and robustly 
    to the global optimum for the given basis.
    
    Logistic functions are used because they provide a smooth transition from 0 to 1, 
    which can better capture sharp transitions, step-like features, or localized 
    changes in the target function compared to global polynomial bases that may 
    oscillate (Runge's phenomenon) or struggle with localized features.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Define 10 fixed logistic basis functions
    # Centers spread across the domain [-2, 2]
    # Scales chosen to provide varying widths for flexibility
    # We use a grid of centers and scales to ensure coverage
    centers = np.linspace(-1.5, 1.5, 10)
    scales = np.array([0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 1.0])
    
    # Compute the basis matrix B where B[i, j] = sigmoid(scales[j] * (x[i] - centers[j]))
    # To avoid numerical issues with large positive/negative exponents, we use a stable sigmoid
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For large z, exp(-z) -> 0, sigmoid -> 1
    # For large -z, exp(-z) -> inf, sigmoid -> 0
    
    # Vectorized computation
    # x is shape (N,), centers is shape (10,)
    # We want shape (N, 10)
    x_expanded = x[:, np.newaxis]
    centers_expanded = centers[np.newaxis, :]
    scales_expanded = scales[np.newaxis, :]
    
    z = scales_expanded * (x_expanded - centers_expanded)
    
    # Stable sigmoid calculation
    # z > 0: 1 / (1 + exp(-z))
    # z <= 0: exp(z) / (1 + exp(z))
    exp_z = np.exp(np.clip(z, -50, 50)) # Clip to prevent overflow/underflow issues in exp
    # Actually, simpler stable sigmoid:
    # np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    # But clipping z in exp is safer for general cases if we don't want branchy logic.
    # Let's use the standard stable form with clipping for safety.
    
    z_clipped = np.clip(z, -50, 50)
    sigmoid_vals = 1.0 / (1.0 + np.exp(-z_clipped))
    
    # Linear combination
    out = sigmoid_vals @ p
    
    # Ensure finite
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        
    return out
