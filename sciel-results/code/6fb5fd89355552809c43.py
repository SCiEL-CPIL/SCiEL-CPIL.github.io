import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a physics-motivated Van der Pol + Duffing basis.
    Basis: x, v, x^3, v*(1-x^2), x^2, x*v, v^3, x^2*v, x*v^2, 1
    The v*(1-x^2) term explicitly encodes the Van der Pol limit cycle mechanism.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Physics-motivated basis
    # p0: x (linear restoring force)
    # p1: v (linear damping)
    # p2: x^3 (Duffing cubic nonlinearity)
    # p3: v*(1-x^2) (Van der Pol damping term)
    # p4: x^2 (quadratic position correction)
    # p5: x*v (cross term)
    # p6: v^3 (cubic velocity)
    # p7: x^2*v (mixed cubic)
    # p8: x*v^2 (mixed cubic)
    # p9: 1 (constant offset)
    
    return (p0 * x + p1 * v + p2 * x3 + p3 * v * (1 - x2) + 
            p4 * x2 + p5 * x * v + p6 * v3 + p7 * x2 * v + 
            p8 * x * v2 + p9)
