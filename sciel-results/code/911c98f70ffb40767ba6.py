"""
Strategy: Local Adaptive Radial Basis Function (RBF) Network.
Instead of a global basis (polynomials/trig), we use 5 Gaussian kernels.
The 10 parameters define 5 centers (p[0:5]) and 5 widths (p[5:10]).
This allows the function to adapt locally to complex shapes.
The coefficients of the RBFs are fixed to 1.0 to avoid scaling issues,
relying on the BFGS optimizer to position and shape the bumps.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    x = np.asarray(x).ravel()
    n = len(x)
    
    # Unpack parameters: 5 centers, 5 widths
    # Centers: p[0], p[1], p[2], p[3], p[4]
    # Widths:   p[5], p[6], p[7], p[8], p[9]
    
    centers = params[0:5]
    widths = params[5:10]
    
    # Ensure widths are strictly positive and not too small to avoid numerical issues
    # If width is close to 0, exp(-x^2/w^2) becomes a spike. 
    # We clamp width to a minimum of 0.01 to prevent division by zero or extreme gradients.
    # We also clamp to a maximum of 10.0 to ensure global coverage if needed.
    safe_widths = np.clip(widths, 0.01, 10.0)
    
    # Initialize output
    result = np.zeros_like(x)
    
    # Compute sum of 5 Gaussian RBFs
    # y(x) = sum_{i=0}^{4} exp(-((x - c_i)^2) / (2 * w_i^2))
    
    # We do this in a loop to be memory efficient and allow early termination if needed
    # Though for n=200, a vectorized approach is fine. 
    # However, the loop allows us to check time if we expand this.
    
    for i in range(5):
        c = centers[i]
        w = safe_widths[i]
        
        # Check deadline
        if time.time() > deadline:
            break
            
        diff = x - c
        # Exponent: -0.5 * (diff / w)^2
        # Using 0.5 * w^2 in denominator
        exponent = -0.5 * (diff * diff) / (w * w)
        
        # Clip exponent to prevent underflow/overflow issues in exp, though exp(-large) -> 0 is safe
        # exp(+large) would be bad, but exponent is always <= 0
        result += np.exp(exponent)
        
    return result
# EVOLVE-BLOCK-END
