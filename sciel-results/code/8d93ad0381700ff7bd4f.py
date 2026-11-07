import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator with Duffing/VdP terms and parametric drive interactions.
    Basis: 1, x, v, x^3, x^2*v, x*v, x*sin(t), v*cos(t), sin(t), cos(t)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    result = (
        p0
        + p1 * x
        + p2 * v
        + p3 * x**3
        + p4 * x**2 * v
        + p5 * x * v
        + p6 * x * np.sin(t)
        + p7 * v * np.cos(t)
        + p8 * np.sin(t)
        + p9 * np.cos(t)
    )
    
    return result
