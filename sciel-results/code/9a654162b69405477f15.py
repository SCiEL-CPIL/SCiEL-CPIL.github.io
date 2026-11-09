import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with time-dependent driving force.
    
    Model:
    a = p0 * sin(p1 * t + p2) + p3 * x + p4 * x^3 + p5 * v + p6 * v^3 + p7 * x * v + p8 * x^2 + p9
    
    Parameters:
    p0: Drive amplitude
    p1: Drive frequency
    p2: Drive phase
    p3: Linear stiffness
    p4: Nonlinear stiffness (Duffing term)
    p5: Linear damping
    p6: Nonlinear damping
    p7: Velocity-position coupling
    p8: Quadratic position term
    p9: Constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Driving force
    drive = p0 * np.sin(p1 * t + p2)
    
    # Nonlinear restoring and damping forces
    restoring = p3 * x + p4 * x**3 + p8 * x**2
    damping = p5 * v + p6 * v**3
    coupling = p7 * x * v
    
    return drive + restoring + damping + coupling + p9
