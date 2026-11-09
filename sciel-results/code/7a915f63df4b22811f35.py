import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Driven Duffing oscillator: a = -gamma*v - alpha*x - beta*x^3 + F*cos(t) + ...
    # Include harmonics of the drive and nonlinear state interactions
    
    # Drive terms: fundamental and 2nd harmonic
    c1 = np.cos(t)
    s1 = np.sin(t)
    c2 = np.cos(2 * t)
    s2 = np.sin(2 * t)
    
    # State variables
    x1 = x
    x3 = x**3
    v1 = v
    xv = x * v
    
    # Additional nonlinear terms
    x2 = x**2
    v2 = v**2
    
    # Combine: 10 basis functions
    return (params[0] * c1 +
            params[1] * s1 +
            params[2] * c2 +
            params[3] * s2 +
            params[4] * x1 +
            params[5] * x3 +
            params[6] * v1 +
            params[7] * xv +
            params[8] * x2 +
            params[9] * v2)
