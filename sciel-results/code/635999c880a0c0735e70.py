"""
Strategy: Adaptive Gaussian Radial Basis Functions (RBF).
Instead of a fixed global polynomial basis, we model the function as a sum of 5
Gaussian bumps. The 10 parameters define the centers (cx, cy) and widths (sigma)
of these bumps. The amplitudes are implicitly determined by the local data fit
within the BFGS optimization loop (since BFGS minimizes MSE, it will effectively
adjust the 'height' contribution by scaling the influence, though here we fix
the amplitude to 1.0 to keep the parameter count at 10, or we can let the
optimizer find the best positions/widths that minimize the error assuming unit
amplitude? No, that's not right.
    
    Wait, if amplitude is fixed to 1, the model is very constrained.
    Let's re-evaluate the parameter allocation.
    If we use 5 RBFs, we need 5 centers (10 params) and 5 widths (5 params). Total 15.
    We only have 10 params.
    
    Alternative: 2 RBFs?
    2 centers (4 params) + 2 widths (2 params) + 2 amplitudes (2 params) = 8 params.
    Plus 2 global linear terms?
    
    Let's try a hybrid: 
    1. Global linear term: a*x + b*y + c (3 params)
    2. 2 Local Gaussian bumps:
       - Center 1: (cx1, cy1) (2 params)
       - Width 1: (s1) (1 param)
       - Amplitude 1: (A1) (1 param)
       - Center 2: (cx2, cy2) (2 params)
       - Width 2: (s2) (1 param)
       - Amplitude 2: (A2) (1 param)
       Total: 3 + 2 + 1 + 1 + 2 + 1 + 1 = 11 params. Too many.
       
    Let's reduce to 1 Gaussian bump + linear?
    Linear: a*x + b*y + c (3)
    Bump: cx, cy, s, A (4)
    Total: 7 params. We have 3 left.
    Maybe another bump?
    Bump 2: cx2, cy2 (2) ... no width/amplitude? Bad.
    
    Let's go back to the "Geometry" idea but simpler.
    How about a Mixture of 2 Gaussians with shared width?
    Params:
    1. cx1
    2. cy1
    3. A1
    4. cx2
    5. cy2
    6. A2
    7. sigma (shared)
    8. linear_x
    9. linear_y
    10. constant
    
    This gives a flexible local-global model.
    f(x,y) = linear_x*x + linear_y*y + constant + A1*exp(-((x-cx1)^2 + (y-cy1)^2)/(2*sigma^2)) + A2*exp(-((x-cx2)^2 + (y-cy2)^2)/(2*sigma^2))
    
    This is a structurally different family (non-linear parameterization of centers/widths) compared to the linear-in-parameters Chebyshev basis.
"""

import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a hybrid Linear + 2 Gaussian RBF model.
    
    Parameters (10 total):
    0: linear_x coefficient
    1: linear_y coefficient
    2: constant term
    3: cx1 (center x of bump 1)
    4: cy1 (center y of bump 1)
    5: A1 (amplitude of bump 1)
    6: cx2 (center x of bump 2)
    7: cy2 (center y of bump 2)
    8: A2 (amplitude of bump 2)
    9: sigma (shared width of bumps)
    
    The non-linearity in parameters (cx, cy, sigma) makes the optimization
    landscape non-convex, but BFGS can often find local minima that correspond
    to good local fits if the initialization (all ones) is not too far off,
    or if the function has localized features.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    lx = params[0]
    ly = params[1]
    c0 = params[2]
    
    cx1 = params[3]
    cy1 = params[4]
    A1 = params[5]
    
    cx2 = params[6]
    cy2 = params[7]
    A2 = params[8]
    
    sigma = params[9]
    
    # Prevent division by zero or extremely small sigma causing overflow
    # If sigma is very small, the Gaussian becomes a delta function.
    # We clamp sigma to a minimum value to ensure numerical stability.
    # A minimum of 0.01 allows for fairly localized bumps.
    sigma_safe = np.maximum(np.abs(sigma), 0.01)
    
    # Calculate squared distances
    # dist1^2 = (x - cx1)^2 + (y - cy1)^2
    dx1 = x - cx1
    dy1 = y - cy1
    dist1_sq = dx1**2 + dy1**2
    
    # dist2^2 = (x - cx2)^2 + (y - cy2)^2
    dx2 = x - cx2
    dy2 = y - cy2
    dist2_sq = dx2**2 + dy2**2
    
    # Calculate Gaussian terms
    # exp(-dist^2 / (2 * sigma^2))
    # To avoid underflow to 0 for large distances (which is fine) or overflow (if sigma is tiny),
    # we rely on sigma_safe.
    term1 = A1 * np.exp(-dist1_sq / (2.0 * sigma_safe**2))
    term2 = A2 * np.exp(-dist2_sq / (2.0 * sigma_safe**2))
    
    # Combine with linear base
    result = lx * x + ly * y + c0 + term1 + term2
    
    # Ensure finite output
    # If any non-finite values appear (e.g., from extreme params), replace with 0
    # or the mean of y to avoid crashing the optimizer with inf/nan.
    # However, the prompt says non-finite error costs 1e30. 
    # It's better to clip the exponent argument to prevent overflow in exp.
    # exp(-large) -> 0. exp(+large) -> inf. 
    # Since dist_sq >= 0 and sigma^2 > 0, the exponent is always <= 0.
    # So exp will never overflow to +inf. It will underflow to 0, which is fine.
    # The only risk is if sigma_safe is 0, but we clamped it.
    
    return result

# EVOLVE-BLOCK-END
