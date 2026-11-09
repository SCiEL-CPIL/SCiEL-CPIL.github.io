"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a Polynomial + Fourier Series basis.
    
    This approach uses a linear combination of polynomial terms and trigonometric
    harmonics. This is effective for smooth functions that may exhibit periodic
    behavior or can be well-approximated by a Fourier series on the bounded domain [-3, 3].
    
    Model:
    y = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 
        + c1*cos(omega*x) + s1*sin(omega*x)
        + c2*cos(2*omega*x) + s2*sin(2*omega*x)
        
    Parameters:
    p0-p4: Polynomial coefficients (constant to quartic)
    p5: Fundamental frequency (omega)
    p6, p7: Amplitudes for 1st harmonic (cos, sin)
    p8, p9: Amplitudes for 2nd harmonic (cos, sin)
    """
    p0, p1, p2, p3, p4, omega, c1, s1, c2, s2 = params
    
    # Polynomial part (degree 4)
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4
    
    # Fourier part
    # Calculate angles
    theta1 = omega * x
    theta2 = 2.0 * omega * x
    
    fourier = (c1 * np.cos(theta1) + s1 * np.sin(theta1) +
               c2 * np.cos(theta2) + s2 * np.sin(theta2))
    
    return poly + fourier
# EVOLVE-BLOCK-END
