"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Cubic Polynomial + 2 Gaussian Bumps.
    
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
    """
    # Extract parameters
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    A1, c1, lw1 = params[4], params[5], params[6]
    A2, c2, lw2 = params[7], params[8], params[9]
    
    # Compute polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Compute Gaussian parts
    # Using exp for width ensures positive width
    # Clip log-width to prevent overflow/underflow issues in exp
    lw1_clipped = np.clip(lw1, -10, 10)
    lw2_clipped = np.clip(lw2, -10, 10)
    
    s1 = np.exp(lw1_clipped)
    s2 = np.exp(lw2_clipped)
    
    g1 = A1 * np.exp(-0.5 * ((x - c1) / s1)**2)
    g2 = A2 * np.exp(-0.5 * ((x - c2) / s2)**2)
    
    return poly + g1 + g2
# EVOLVE-BLOCK-END
