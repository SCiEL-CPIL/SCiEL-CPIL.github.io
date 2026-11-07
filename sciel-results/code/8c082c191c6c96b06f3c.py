import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with time-varying coefficients.
    
    Structural change: Use a low-order Fourier series in t to modulate the
    coefficients of the state variables x and v. This captures periodic
    changes in effective stiffness and damping.
    
    Parameters:
    p0, p1, p2: Coefficients for constant term (a0 + a1*sin(t) + a2*cos(t))
    p3, p4, p5: Coefficients for x term (b0 + b1*sin(t) + b2*cos(t))
    p6, p7, p8: Coefficients for v term (c0 + c1*sin(t) + c2*cos(t))
    p9: Coefficient for x^3 term (constant nonlinear stiffness)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    x2 = x * x
    x3 = x2 * x
    
    # Time-varying coefficients
    a_t = p0 + p1 * sin_t + p2 * cos_t
    b_t = p3 + p4 * sin_t + p5 * cos_t
    c_t = p6 + p7 * sin_t + p8 * cos_t
    
    # Combine
    return a_t + b_t * x + c_t * v + p9 * x3
