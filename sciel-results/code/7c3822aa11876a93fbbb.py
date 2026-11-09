"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: low-degree polynomial + tunable sigmoid + rational term.
    
    Model:
    y(x) = offset
         + (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
         + s0 * sigmoid(s1 * x + s2)
         + (r0 + r1*x + r2*x^2) / (1.0 + b1*x + b2*x^2)
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 parameters
            [a0, a1, a2, a3, a4, s0, s1, s2, r0, r1, r2, b1, b2, offset]
            Wait, that's 14. We need exactly 10.
            
    Let me restructure to fit exactly 10 parameters:
    y(x) = offset
         + (a0 + a1*x + a2*x^2)  # quadratic polynomial (3 params: a0,a1,a2)
         + s0 * sigmoid(s1 * x + s2)  # sigmoid term (3 params: s0,s1,s2)
         + (r0 + r1*x) / (1.0 + b1*x)  # rational term (3 params: r0,r1,b1)
         + offset  # 1 param
    
    Total: 3 + 3 + 3 + 1 = 10 params.
    """
    # Unpack 10 parameters
    a0, a1, a2, s0, s1, s2, r0, r1, b1, offset = params
    
    # Quadratic polynomial part
    poly = a0 + x * (a1 + x * a2)
    
    # Sigmoid part: s0 * sigmoid(s1*x + s2)
    # Use stable sigmoid computation
    z = s1 * x + s2
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -500, 500)
    sig = 1.0 / (1.0 + np.exp(-z_clipped))
    sig_term = s0 * sig
    
    # Rational part: (r0 + r1*x) / (1.0 + b1*x)
    # Safe division
    den = 1.0 + b1 * x
    # Avoid division by zero
    den_safe = np.where(np.abs(den) < 1e-10, 1e-10 * np.sign(den + 1e-300), den)
    rat_term = (r0 + r1 * x) / den_safe
    
    result = offset + poly + sig_term + rat_term
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
