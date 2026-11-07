import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed nonlinear oscillator model (Duffing + Van der Pol structure)
    with smooth nonlinear corrections.
    
    The model captures:
    - Duffing restoring force: linear + cubic in x
    - Van der Pol damping: linear + amplitude-dependent (x^2 - c)*v
    - Smooth nonlinear corrections via tanh and sin for bounded behavior
    
    Parameters (10):
      p0: linear restoring coefficient (omega^2)
      p1: cubic restoring coefficient (Duffing beta)
      p2: linear damping coefficient (gamma)
      p3: Van der Pol amplitude parameter
      p4: x^2 * v interaction
      p5: x * v interaction
      p6: tanh(x) * v correction
      p7: sin(x) * v correction
      p8: tanh(x) correction
      p9: constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Duffing restoring force: -p0*x - p1*x^3
    restoring = -p0 * x - p1 * x**3

    # Van der Pol damping: -p2*v - p3*(x^2 - 1)*v
    # Standard Van der Pol: dv/dt = -x + (mu - x^2)*v, so damping term is -(mu - x^2)*v = -mu*v + x^2*v
    # Here we use: -p2*v - p3*(x^2 - 1)*v = -p2*v - p3*x^2*v + p3*v = (p3 - p2)*v - p3*x^2*v
    # Let's keep it explicit:
    vdp_damping = -p2 * v - p3 * (x**2 - 1.0) * v

    # Additional interaction terms
    x2v = p4 * x**2 * v
    xv = p5 * x * v

    # Smooth nonlinear corrections (bounded, helps with global structure)
    tanh_x_v = p6 * np.tanh(x) * v
    sin_x_v = p7 * np.sin(x) * v
    tanh_x = p8 * np.tanh(x)

    # Constant offset
    offset = p9

    # Combine all terms
    result = restoring + vdp_damping + x2v + xv + tanh_x_v + sin_x_v + tanh_x + offset

    return result
