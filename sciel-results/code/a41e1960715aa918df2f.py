"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Cubic Polynomial + 2 Gaussian Bumps with numerically stable computation.
    
    y(x) = p0 + p1*x + p2*x^2 + p3*x^3 
           + p4 * exp(-0.5 * ((x - p5) / exp(p6))^2)
           + p7 * exp(-0.5 * ((x - p8) / exp(p9))^2)
           
    Parameters:
    p0, p1, p2, p3: Polynomial coefficients (constant to cubic)
    p4: Amplitude of Gaussian 1
    p5: Center of Gaussian 1
    p6: Log-width of Gaussian 1 (width = exp(p6))
    p7: Amplitude of Gaussian 2
    p8: Center of Gaussian 2
    p9: Log-width of Gaussian 2 (width = exp(p9))
    
    Uses numerically stable exp to prevent overflow/underflow.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    A1, c1, lw1 = p[4], p[5], p[6]
    A2, c2, lw2 = p[7], p[8], p[9]
    
    # Compute polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Compute Gaussian parts with numerically stable width
    # Clip log-width to prevent overflow/underflow in exp
    lw1_clipped = np.clip(lw1, -10, 10)
    lw2_clipped = np.clip(lw2, -10, 10)
    
    s1 = np.exp(lw1_clipped)
    s2 = np.exp(lw2_clipped)
    
    # Compute exponents for Gaussian bumps
    # exp_arg = -0.5 * ((x - c) / s)^2
    # To prevent underflow to 0 for very negative exponents, we clip
    # but for very positive (which can't happen since it's -0.5 * square), no issue
    # The main risk is if s is very small, making (x-c)/s very large, so square is huge
    # exp of a very negative number is 0, which is fine
    # Let's compute the argument and clip it to avoid any potential issues
    # Actually, exp(-large_negative) = 0, which is safe.
    # exp(-large_positive) would be overflow, but the argument is always <= 0.
    # So we just need to ensure we don't have NaN from inf/inf or similar.
    
    arg1 = -0.5 * ((x - c1) / s1) ** 2
    arg2 = -0.5 * ((x - c2) / s2) ** 2
    
    # Clip arguments to a reasonable range to prevent any numerical issues
    # exp(-1000) is essentially 0, exp(0) is 1
    # We can safely clip to [-100, 0] or similar
    arg1 = np.clip(arg1, -100, 0)
    arg2 = np.clip(arg2, -100, 0)
    
    g1 = A1 * np.exp(arg1)
    g2 = A2 * np.exp(arg2)
    
    result = poly + g1 + g2
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        # This shouldn't happen for a single evaluation, but as a guard
        return result
    
    return result
# EVOLVE-BLOCK-END
