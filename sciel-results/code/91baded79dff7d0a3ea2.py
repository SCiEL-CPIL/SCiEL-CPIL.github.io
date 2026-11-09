"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model with 10 parameters.
    
    Basis functions:
    b0 = x
    b1 = 1
    b2 = x * tanh(p4 * x)
    b3 = tanh(p4 * x)
    b4 = x * sin(p5 * x)
    b5 = cos(p5 * x)
    b6 = exp(-p7 * x^2)
    
    Parameters:
    p[0]: coeff for b2
    p[1]: coeff for b3
    p[2]: coeff for b4
    p[3]: coeff for b5
    p[4]: freq/scale for tanh
    p[5]: freq for sin/cos
    p[6]: coeff for b6
    p[7]: width for gaussian
    p[8]: coeff for b0 (linear)
    p[9]: coeff for b1 (constant)
    """
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill

    p = params
    
    # Ensure parameters are finite to avoid NaNs in optimization
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    x2 = x * x
    
    # Calculate basis functions
    # p[4] is the scale for tanh. We clip it to avoid extreme values that might cause issues
    # though tanh is stable, large args just go to +/-1.
    tanh_arg = p[4] * x
    b2 = x * np.tanh(tanh_arg)
    b3 = np.tanh(tanh_arg)
    
    # p[5] is the frequency for sin/cos
    sin_arg = p[5] * x
    b4 = x * np.sin(sin_arg)
    b5 = np.cos(sin_arg)
    
    # p[7] is the width for Gaussian. Ensure it's positive to avoid explosion?
    # exp(-negative * x^2) = exp(positive * x^2) which explodes for |x|>0.
    # BFGS might try negative values. We use abs(p[7]) to ensure stability.
    gauss_width = abs(p[7])
    b6 = np.exp(-gauss_width * x2)
    
    b0 = x
    b1 = np.ones_like(x)
    
    # Combine
    y = (
        p[0] * b2 +
        p[1] * b3 +
        p[2] * b4 +
        p[3] * b5 +
        p[6] * b6 +
        p[8] * b0 +
        p[9] * b1
    )
    
    # Check for non-finite results
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
