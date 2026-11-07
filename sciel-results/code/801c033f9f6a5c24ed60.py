import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sine-based nonlinear oscillator model with phase-dependent modulation.
    
    Structure: a = f(x, v) using trigonometric functions for smoother
    nonlinear behavior and better generalization.
    
    Parameters (10):
      p0: linear restoring coefficient
      p1: sine(x) restoring amplitude
      p2: cos(x) * v coupling
      p3: sine(v) damping
      p4: cos(v) * x coupling
      p5: x*v product term
      p6: tanh(x) * v nonlinear damping
      p7: tanh(v) * x nonlinear restoring
      p8: x^2 * v^2 cross term
      p9: bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    sx = np.sin(x)
    cx = np.cos(x)
    sv = np.sin(v)
    cv = np.cos(v)
    tx = np.tanh(x)
    tv = np.tanh(v)
    
    x2 = x * x
    v2 = v * v
    
    # Restoring forces
    restore = p0 * x + p1 * sx + p7 * tv * x
    
    # Damping and velocity-dependent terms
    damping = p3 * sv + p6 * tx * v
    
    # Coupling terms
    coupling = p2 * cx * v + p4 * cv * x + p5 * x * v
    
    # Higher-order cross term
    cross = p8 * x2 * v2
    
    # Constant
    offset = p9 * np.ones_like(x)
    
    result = restore + damping + coupling + cross + offset
    
    return result
