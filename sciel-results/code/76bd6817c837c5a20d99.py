import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol-inspired polynomial basis for nonlinear oscillator dynamics.
    
    This model uses a physics-informed polynomial basis that captures:
    - Linear restoring force (x)
    - Linear damping (v)
    - Cubic nonlinearity (x^3) for Duffing-like hardening
    - Amplitude-dependent damping terms (v^2, x^2*v, x*v^2)
    - Higher-order corrections (x^5, v^3, x^4, x^2*v^2)
    
    Basis functions (10 parameters):
    p0: 1 (constant)
    p1: x (linear restoring)
    p2: v (linear damping)
    p3: x^2 (quadratic position)
    p4: x^3 (cubic restoring)
    p5: v^2 (quadratic velocity)
    p6: x*v (cross term)
    p7: x^4 (quartic position)
    p8: x^2*v (amplitude-dependent damping)
    p9: x*v^2 (velocity-dependent coupling)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute powers efficiently
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    v2 = v * v
    v3 = v2 * v
    
    # Construct basis terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = v
    b3 = x2
    b4 = x3
    b5 = v2
    b6 = x * v
    b7 = x4
    b8 = x2 * v
    b9 = x * v2
    
    # Linear combination
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return result
