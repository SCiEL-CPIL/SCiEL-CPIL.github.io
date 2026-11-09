import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Core oscillator terms: driving force, damping, and restoring force
    # Drive: cos(t) is typical for forced oscillators
    c1 = np.cos(t)
    # Damping: linear in velocity
    v1 = v
    # Restoring force: linear and cubic (Duffing oscillator)
    x1 = x
    x3 = x**3
    
    # Additional basis functions to capture potential nuances
    # Sinusoidal drive component
    s1 = np.sin(t)
    # Quadratic nonlinearity in x (can be represented via x^3 and x, but adding for flexibility)
    x2 = x**2
    # Quadratic nonlinearity in v
    v2 = v**2
    
    # Constant term (bias)
    ones = np.ones_like(t)
    
    # Combine all terms with fitted parameters
    return (params[0] * c1 + 
            params[1] * s1 + 
            params[2] * v1 + 
            params[3] * v2 + 
            params[4] * x1 + 
            params[5] * x2 + 
            params[6] * x3 + 
            params[7] * (x * v) +      # Interaction term
            params[8] * (v**3) +        # Cubic damping
            params[9] * ones)
