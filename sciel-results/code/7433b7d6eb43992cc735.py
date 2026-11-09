import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Hybrid Adaptive Local-Global Basis.
    Uses 4 parameters to define a single adaptive Gaussian RBF (center, scale, amplitude)
    and 6 parameters for a low-order polynomial trend (1, x, y, x^2, y^2, xy).
    
    This is materially different from the parent (fixed RBFs, linear in params) because:
    1. The RBF parameters are non-linear in the raw params (tanh, exp).
    2. The optimization problem is non-convex; BFGS must search for the optimal
       location/scale of the local feature, not just weight fixed features.
    3. The polynomial part handles global smooth trends, allowing the RBF to focus
       on local deviations.
    """
    start_time = time.time()
    
    # Safety deadline guard
    if time.time() - start_time > 110:
        # Return a simple fallback if we are running out of time (unlikely for this simple calc)
        return np.zeros_like(x)

    # Ensure params is 1D and length 10
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    # --- Part 1: Adaptive Gaussian RBF ---
    # Params 0-3
    cx = np.tanh(p[0])
    cy = np.tanh(p[1])
    # Scale: exp(p[2]), clamped to prevent numerical issues
    # If p[2] is very negative, sigma is small (sharp). If very positive, sigma is large (flat).
    # Clamp sigma to [0.05, 2.0] to keep features resolvable and bounded.
    sigma = np.exp(np.clip(p[2], -3.0, 1.0)) # exp(-3)~0.05, exp(1)~2.7
    amplitude = p[3]
    
    # Calculate RBF term
    dx = x - cx
    dy = y - cy
    dist_sq = dx**2 + dy**2
    rbf_term = amplitude * np.exp(-dist_sq / (2.0 * sigma**2))
    
    # --- Part 2: Polynomial Trend ---
    # Params 4-9
    # Basis: 1, x, y, x^2, y^2, xy
    poly_term = (p[4] + 
                 p[5] * x + 
                 p[6] * y + 
                 p[7] * x**2 + 
                 p[8] * y**2 + 
                 p[9] * x * y)
                 
    pred = rbf_term + poly_term
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
