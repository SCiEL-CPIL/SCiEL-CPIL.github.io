import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model:
    a = offset + (c1*v + c2*v^2 + c3*v^3) + (k1*x + k2*x^2 + k3*x^3) + A*sin(w*t + phi)
    """
    p = params
    # Damping terms
    damp = p[1] * v + p[2] * v**2 + p[3] * v**3
    # Stiffness terms
    stiff = p[4] * x + p[5] * x**2 + p[6] * x**3
    # Driving force
    drive = p[7] * np.sin(p[8] * t + p[9])
    
    return p[0] + damp + stiff + drive
