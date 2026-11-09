import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Generic polynomial basis model for the oscillator.
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p[0]: Intercept
        p[1]: Coeff for centered time (t-40)
        p[2]: Coeff for position x
        p[3]: Coeff for velocity v
        p[4]: Coeff for x^2
        p[5]: Coeff for v^2
        p[6]: Coeff for x*v
        p[7]: Coeff for v^3
        p[8]: Coeff for x*v^2
        p[9]: Coeff for (t-40)^2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Centered time to improve conditioning
    tc = t - 40.0
    
    # Construct basis terms
    term_0 = 1.0
    term_1 = tc
    term_2 = x
    term_3 = v
    term_4 = x**2
    term_5 = v**2
    term_6 = x * v
    term_7 = v**3
    term_8 = x * v**2
    term_9 = tc**2
    
    # Combine linearly
    return p0*term_0 + p1*term_1 + p2*term_2 + p3*term_3 + p4*term_4 + p5*term_5 + p6*term_6 + p7*term_7 + p8*term_8 + p9*term_9
