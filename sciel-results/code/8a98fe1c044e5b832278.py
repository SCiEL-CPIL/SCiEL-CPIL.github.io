import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Driven Duffing oscillator with nonlinear damping and drive harmonics
    # Target is likely acceleration: a = F*cos(w*t) - gamma*v - alpha*x - beta*x^3 + ...
    
    # Drive terms: fundamental and 2nd harmonic
    c1 = np.cos(t)
    s1 = np.sin(t)
    c2 = np.cos(2 * t)
    s2 = np.sin(2 * t)
    
    # State variables and nonlinear terms
    x1 = x
    x3 = x**3
    v1 = v
    xv = x * v
    x2v = x**2 * v
    
    # Constant term
    ones = np.ones_like(t)
    
    # Combine 10 basis functions with fitted parameters
    # Order: c1, s1, c2, s2, x1, x3, v1, xv, x2v, ones
    return (params[0] * c1 +
            params[1] * s1 +
            params[2] * c2 +
            params[3] * s2 +
            params[4] * x1 +
            params[5] * x3 +
            params[6] * v1 +
            params[7] * xv +
            params[8] * x2v +
            params[9] * ones)
