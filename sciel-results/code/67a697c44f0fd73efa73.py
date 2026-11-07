import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Basis: 1, x, x^3, v, x^2*v, sin(omega*t), cos(omega*t), x*v, x^2
    where omega is a fitted parameter (params[5]).
    This covers Duffing, Van der Pol, and forced oscillation with fitted frequency.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x
    x2v = x2 * v
    xv = x * v

    # Fitted frequency
    omega = p5
    
    # Sinusoidal driving terms with fitted frequency
    st = np.sin(omega * t)
    ct = np.cos(omega * t)

    return (
        p0
        + p1 * x
        + p2 * x3
        + p3 * v
        + p4 * x2v
        + p6 * st
        + p7 * ct
        + p8 * xv
        + p9 * x2
    )
