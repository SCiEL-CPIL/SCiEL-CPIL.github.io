"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a mixture of nonlinear basis functions.
    
    Parameters (length 10):
    p0, p1, p2: Gaussian term p0 * exp(-p1 * (x - p2)^2)
                p0: amplitude, p1: inverse variance (width control), p2: center
    p3, p4, p5: Sine term p3 * sin(p4 * x + p5)
                p3: amplitude, p4: frequency, p5: phase
    p6, p7:     Exponential decay term p6 * exp(-p7 * x)
                p6: amplitude, p7: decay rate
    p8, p9:     Linear term p8 * x + p9
                p8: slope, p9: intercept
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Clip parameters to prevent numerical instability (overflow in exp, etc.)
    # p1, p7 are in exponents, so keep them bounded.
    # p4 is frequency, keep it bounded to avoid aliasing/oscillation issues.
    p1 = np.clip(p1, -10.0, 10.0)
    p2 = np.clip(p2, -10.0, 10.0)
    p4 = np.clip(p4, -10.0, 10.0)
    p5 = np.clip(p5, -10.0, 10.0)
    p7 = np.clip(p7, -10.0, 10.0)
    
    # Calculate basis functions
    # 1. Gaussian
    # Ensure p1 is positive enough for a standard Gaussian shape, 
    # but allowing negative p1 allows for "anti-Gaussian" or inverted shapes if needed, 
    # though typically we want a bump. 
    # Actually, exp(-p1 * (x-p2)^2) with negative p1 blows up. 
    # Let's force p1 to be positive for the Gaussian width to be stable.
    p1_safe = np.maximum(p1, 1e-6)
    gauss = p0 * np.exp(-p1_safe * (x - p2) ** 2)
    
    # 2. Sine
    sine = p3 * np.sin(p4 * x + p5)
    
    # 3. Exponential decay
    # Force p7 to be positive for decay? If p7 is negative, it grows. 
    # exp(-p7*x) can be large if p7 is negative and x is positive.
    # Let's allow it but clip the result if it gets too big? 
    # Better: just clip p7 to be non-negative for decay, or handle both sides.
    # Given x in [-2, 2], exp(10*2) = exp(20) ~ 4.8e8. This is manageable in float64.
    # But gradients might be huge. 
    # Let's stick to the clip range [-10, 10] which is safe for exp in float64.
    exp_term = p6 * np.exp(-p7 * x)
    
    # 4. Linear
    linear = p8 * x + p9
    
    y = gauss + sine + exp_term + linear
    
    return y
# EVOLVE-BLOCK-END
