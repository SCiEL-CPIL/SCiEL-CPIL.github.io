import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses a smooth basis of 10 terms to model nonlinear dynamics:
    1.  Constant offset
    2.  Linear position (x)
    3.  Linear velocity (v)
    4.  Quadratic position (x^2)
    5.  Cubic position (x^3) - Duffing nonlinearity
    6.  State-dependent damping (x^2 * v) - Van der Pol nonlinearity
    7.  Cubic velocity (v^3) - High-order damping
    8.  Cross-coupling (x * v)
    9.  Sine driving (sin(t))
    10. Cosine driving (cos(t))
    """
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    return (
        params[0]
        + params[1] * x
        + params[2] * v
        + params[3] * x2
        + params[4] * x3
        + params[5] * x2 * v
        + params[6] * v3
        + params[7] * x * v
        + params[8] * np.sin(t)
        + params[9] * np.cos(t)
    )
