"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Basis Expansion for Symbolic Regression.
    
    Combines the local approximation power of Gaussian kernels (from current program)
    with the global functional coverage of a mixed polynomial/trig/exp basis 
    (from inspirations). 
    
    Rationale:
    - The previous Gaussian-only model scored ~169, suggesting the target function 
      has significant global structure (polynomial, trig, or exp trends) that 
      local Gaussians struggle to capture efficiently with only 10 parameters.
    - The previous mixed basis models scored poorly (~3e6), likely due to 
      ill-conditioning from terms like exp(x) or high-degree polynomials, or 
      simply poor fit to the specific target.
    - This hybrid uses 5 Gaussian kernels for local flexibility and 5 global 
      basis functions (1, x, x^2, sin(x), cos(x)) for capturing overall trends.
    - The basis is well-conditioned: 
        * Gaussians are smooth and localized.
        * Polynomials up to x^2 are stable on [0.1, 6].
        * Trig functions are bounded.
        * No exp(x) or 1/x to avoid extreme values/ill-conditioning.
    
    Model:
    f(x) = sum_{i=0}^{4} p_i * exp(-0.5 * ((x - c_i) / sigma)^2) 
           + p_5 * 1 
           + p_6 * x 
           + p_7 * x^2 
           + p_8 * sin(x) 
           + p_9 * cos(x)
    
    Since the model is linear in p_i, BFGS will converge to the optimal 
    least-squares solution for this basis.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback to simple linear if running out of time
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    
    # --- Part 1: Local Gaussian Components (5 kernels) ---
    # Centers spread across the domain [0.1, 6]
    # Using 5 centers to leave room for global terms
    centers = np.array([0.5, 1.8, 3.1, 4.4, 5.7])
    # Fixed standard deviation, wide enough to overlap but distinct
    sigma = 0.7
    
    diffs = x_arr[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diffs / sigma) ** 2)  # Shape: (N, 5)
    
    # Linear combination of Gaussians
    local_part = gaussians @ p[0:5]
    
    # --- Part 2: Global Basis Components (5 terms) ---
    # 1, x, x^2, sin(x), cos(x)
    # These capture polynomial and periodic trends
    global_basis = np.column_stack((
        np.ones_like(x_arr),
        x_arr,
        x_arr ** 2,
        np.sin(x_arr),
        np.cos(x_arr)
    ))  # Shape: (N, 5)
    
    global_part = global_basis @ p[5:10]
    
    # Total prediction
    result = local_part + global_part
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to simple linear if numerical issues arise
        result = p[0] * x_arr + p[1]
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
