"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear combination of diverse global basis functions 
    (polynomials, trigonometric, exponential, rational) plus local Gaussian bumps.
    
    The first 8 parameters (params[0:8]) define a global function using a rich 
    basis set: constant, linear, quadratic, cubic, sine, cosine, exp(x), exp(-x).
    These are scaled/centered to improve conditioning.
    
    The last 4 parameters (params[8:12]) define two local Gaussian bumps:
    params[8]: amplitude w1
    params[9]: center c1
    params[10]: amplitude w2
    params[11]: center c2
    
    The widths of the Gaussians are fixed to provide local resolution.
    This hybrid approach allows the model to capture both global trends and 
    local features (peaks, dips, sharp transitions) that a pure global basis 
    might miss.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack parameters
    # Global basis coefficients
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    p4, p5, p6, p7 = params[4], params[5], params[6], params[7]
    
    # Gaussian parameters
    w1, c1, w2, c2 = params[8], params[9], params[10], params[11]
    
    # Define global basis functions, centered/scaled for better conditioning
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic (centered: mean of x^2 on [-2,2] is 4/3)
    b2 = x**2 - 4.0/3.0
    
    # 4. Cubic (odd, mean 0)
    b3 = x**3
    
    # 5. Sine
    b4 = np.sin(x)
    
    # 6. Cosine (centered: mean of cos(x) on [-2,2] is sin(2)/2 ~ 0.4546)
    b5 = np.cos(x) - 0.4546
    
    # 7. Exponential growth (centered: mean of e^x on [-2,2] is (e^2-e^-2)/4 ~ 1.813)
    b6 = np.exp(x) - 1.813
    
    # 8. Exponential decay (centered: mean of e^-x on [-2,2] is same as e^x)
    b7 = np.exp(-x) - 1.813
    
    # Global part
    global_part = (
        p0 * b0 +
        p1 * b1 +
        p2 * b2 +
        p3 * b3 +
        p4 * b4 +
        p5 * b5 +
        p6 * b6 +
        p7 * b7
    )
    
    # Local Gaussian bumps
    # Fixed width sigma = 0.5, so sigma^2 = 0.25
    # exp(-(x-c)^2 / (2*sigma^2)) = exp(-(x-c)^2 / 0.5) = exp(-2*(x-c)^2)
    
    # Clip centers to reasonable range to avoid numerical issues
    c1_clipped = np.clip(c1, -2.5, 2.5)
    c2_clipped = np.clip(c2, -2.5, 2.5)
    
    diff1 = x - c1_clipped
    diff2 = x - c2_clipped
    
    # Ensure exponent is negative to avoid overflow
    # diff^2 is always non-negative, so -2*diff^2 is always non-positive
    g1 = np.exp(-2.0 * diff1 * diff1)
    g2 = np.exp(-2.0 * diff2 * diff2)
    
    gaussian_part = w1 * g1 + w2 * g2
    
    result = global_part + gaussian_part
    
    # Safety: replace non-finite values with a large number to penalize
    result = np.where(np.isfinite(result), result, 1e6)
    
    return result
# EVOLVE-BLOCK-END
