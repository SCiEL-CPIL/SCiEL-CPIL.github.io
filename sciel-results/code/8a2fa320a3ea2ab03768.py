import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator with Duffing and Van der Pol terms, 
    plus sinusoidal driving.
    Form: p0 + p1*x + p2*x^2 + p3*x^3 + p4*v + p5*x*v + p6*x^2*v + p7*v^2 + p8*sin(t) + p9*cos(t)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and trig functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    x2v = x2 * v
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    result = (
        p0 
        + p1 * x 
        + p2 * x2 
        + p3 * x3 
        + p4 * v 
        + p5 * xv 
        + p6 * x2v 
        + p7 * v2 
        + p8 * sin_t 
        + p9 * cos_t
    )
    
    return result
