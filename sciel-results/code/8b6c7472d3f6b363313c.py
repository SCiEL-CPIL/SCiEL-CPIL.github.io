import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration: Duffing + Van der Pol + sinusoidal drive.
    a = -omega0^2*x - beta*x^3 - mu*v*(1-x^2) + F0*cos(omega*t)
    Expanded with 10 free constants for BFGS fitting.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params

    # Center time for numerical conditioning
    t_c = t - 40.0

    # State powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v

    # Sinusoidal drive with two frequencies
    drive1 = np.sin(c8 * t_c)
    drive2 = np.cos(c8 * t_c)

    # Physically-motivated basis:
    # c0: constant offset
    # c1*x: linear spring (Duffing linear part)
    # c2*x^3: Duffing nonlinearity
    # c3*v: linear damping
    # c4*v^3: Van der Pol nonlinear damping
    # c5*x^2*v: state-dependent damping
    # c6*x*v: cross damping
    # c7*drive1: sinusoidal drive (sin)
    # c8 is frequency (used above), so c7 multiplies sin
    # c9*drive2: sinusoidal drive (cos)
    #
    # Reorganize: use c8 as frequency, and allocate remaining for terms
    # Since c8 is inside sin/cos, we have 9 other coefficients + c8 = 10 total

    result = (c0
              + c1 * x
              + c2 * x3
              + c3 * v
              + c4 * v3
              + c5 * x2 * v
              + c6 * x * v
              + c7 * drive1
              + c9 * drive2)

    return result
