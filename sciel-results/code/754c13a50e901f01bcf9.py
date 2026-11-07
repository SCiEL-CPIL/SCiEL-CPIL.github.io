import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a mixed polynomial + tanh basis.
    The tanh terms provide smooth nonlinear saturation that better captures
    oscillator dynamics (limit cycles, amplitude-dependent behavior) than
    pure polynomials.
    
    Basis terms:
    1. 1 (bias)
    2. x (linear restoring)
    3. v (linear damping)
    4. tanh(x) (saturated restoring)
    5. tanh(v) (saturated damping)
    6. x * tanh(x) (nonlinear restoring correction)
    7. v * tanh(v) (nonlinear damping correction)
    8. x * v (cross coupling)
    9. x^2 (quadratic position)
    10. v^2 (quadratic velocity)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute tanh features
    tx = np.tanh(x)
    tv = np.tanh(v)
    
    # Basis terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = v
    b3 = tx
    b4 = tv
    b5 = x * tx
    b6 = v * tv
    b7 = x * v
    b8 = x * x
    b9 = v * v
    
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return result
