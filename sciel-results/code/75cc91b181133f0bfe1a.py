import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using Fourier series in time and polynomial state terms.
    
    Model:
    a = p1*sin(t) + p2*cos(t) + p3*sin(2t) + p4*cos(2t) 
      + p5*x + p6*v + p7*x^2 + p8*v^2 + p9*x*sin(t) + p10*v*cos(t)
    
    Parameters:
    p1-p4: Coefficients for 1st and 2nd harmonic Fourier series in time
    p5-p8: Coefficients for linear and quadratic state terms
    p9-p10: Coefficients for state-time interaction terms
    """
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    p4 = params[3]
    p5 = params[4]
    p6 = params[5]
    p7 = params[6]
    p8 = params[7]
    p9 = params[8]
    p10 = params[9]
    
    # Fourier components in time
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_2t = np.sin(2 * t)
    cos_2t = np.cos(2 * t)
    
    # State components
    x2 = x * x
    v2 = v * v
    
    # Interaction terms
    x_sin_t = x * sin_t
    v_cos_t = v * cos_t
    
    # Construct the equation
    result = (
        p1 * sin_t + 
        p2 * cos_t + 
        p3 * sin_2t + 
        p4 * cos_2t + 
        p5 * x + 
        p6 * v + 
        p7 * x2 + 
        p8 * v2 + 
        p9 * x_sin_t + 
        p10 * v_cos_t
    )
    
    return result
