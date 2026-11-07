import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a well-conditioned basis of
    normalized polynomial and trigonometric terms.
    
    The basis is designed for BFGS convergence:
    - All terms are bounded in the training domain
    - Terms are orthogonal-ish to reduce collinearity
    - Mix of odd/even symmetries to capture general nonlinear behavior
    
    x range: [-0.5728, 0.5275] -> normalize to [-1, 1]
    v range: [-0.1754, 0.2821] -> normalize to [-1, 1]
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to roughly [-1, 1] for better conditioning
    xn = x / 0.55  # approx center of range
    vn = v / 0.23  # approx center of range

    # Basis: mix of polynomial and trig, all bounded
    # Odd in x (restoring force candidates)
    b0 = xn
    b1 = xn * xn * xn  # cubic
    b2 = np.sin(xn)
    
    # Odd in v (damping candidates)
    b3 = vn
    b4 = vn * vn * vn  # cubic
    b5 = np.sin(vn)
    
    # Coupling terms
    b6 = xn * vn
    b7 = np.sin(xn) * np.sin(vn)
    
    # Even terms / bias-like
    b8 = xn * xn - 0.5  # shifted quadratic, mean-zero-ish
    b9 = 1.0  # bias

    return (
        p0 * b0
        + p1 * b1
        + p2 * b2
        + p3 * b3
        + p4 * b4
        + p5 * b5
        + p6 * b6
        + p7 * b7
        + p8 * b8
        + p9 * b9
    )
