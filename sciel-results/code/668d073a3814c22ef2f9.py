import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol-inspired nonlinear oscillator acceleration model.
    
    Structure: a = f(x, v) with amplitude-dependent damping.
    The key feature is that damping changes sign with amplitude,
    creating a limit cycle: energy injection at small amplitudes,
    dissipation at large amplitudes.
    
    Parameters (10):
      p0: linear restoring (omega^2)
      p1: cubic restoring (Duffing beta)
      p2: linear damping (base gamma)
      p3: amplitude-dependent damping (Van der Pol: -p2*v + p3*x^2*v)
      p4: x*v parametric coupling
      p5: cubic damping (v^3)
      p6: v^2*x velocity-dependent stiffness
      p7: x^2*v^2 cross-damping
      p8: x^5 higher-order restoring
      p9: bias / offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    v2 = v * v
    v3 = v2 * v
    
    # Restoring forces (potential derivatives)
    restore = p0 * x + p1 * x3 + p8 * x5
    
    # Damping: Van der Pol structure => (p2 + p3*x2) * v
    # At small x: damping ~ p2*v (could be negative = energy injection)
    # At large x: damping ~ (p2 + p3*x2)*v (positive = dissipation)
    damping = (p2 + p3 * x2) * v + p5 * v3
    
    # Cross/coupling terms
    coupling = p4 * x * v + p6 * v2 * x + p7 * x2 * v2
    
    # Constant
    offset = p9 * np.ones_like(x)
    
    result = restore + damping + coupling + offset
    
    return result
