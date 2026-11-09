"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Polynomial + Fourier Series with Learnable Frequency.
    
    This model combines a low-degree polynomial with a Fourier series to capture
    both global trends and periodic/oscillatory behaviors. The frequency of the
    Fourier terms is a learnable parameter, allowing the model to adapt to the
    period of the target function.
    
    y(x) = p0 + p1*x + p2*x^2 
           + p4*sin(p3*x) + p5*cos(p3*x)
           + p6*sin(2*p3*x) + p7*cos(2*p3*x)
           + p8*sin(3*p3*x) + p9*cos(3*p3*x)
           
    Parameters:
    p0, p1, p2: Polynomial coefficients (constant, linear, quadratic)
    p3: Frequency omega
    p4, p5: Amplitudes for 1st harmonic (sin, cos)
    p6, p7: Amplitudes for 2nd harmonic (sin, cos)
    p8, p9: Amplitudes for 3rd harmonic (sin, cos)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2 = p[0], p[1], p[2]
    omega = p[3]
    
    # Compute polynomial part
    poly = p0 + p1 * x + p2 * x**2
    
    # Compute Fourier terms
    # Use absolute value for frequency to avoid sign ambiguity in harmonics?
    # No, sin(-w x) = -sin(w x), cos(-w x) = cos(w x). 
    # The amplitudes can absorb the sign change for sin, but cos stays same.
    # It's safer to just use omega directly.
    
    # To ensure numerical stability for large x*omega, we rely on sin/cos being bounded.
    
    term1_s = np.sin(omega * x)
    term1_c = np.cos(omega * x)
    
    term2_s = np.sin(2.0 * omega * x)
    term2_c = np.cos(2.0 * omega * x)
    
    term3_s = np.sin(3.0 * omega * x)
    term3_c = np.cos(3.0 * omega * x)
    
    # Combine
    y = poly + p[4] * term1_s + p[5] * term1_c + \
            p[6] * term2_s + p[7] * term2_c + \
            p[8] * term3_s + p[9] * term3_c
            
    return y
# EVOLVE-BLOCK-END
