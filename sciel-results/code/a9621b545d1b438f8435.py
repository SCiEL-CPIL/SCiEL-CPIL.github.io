import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a normalized, bounded feature basis.
    
    Inputs are normalized to unit scale to improve BFGS conditioning from the
    all-ones start. The feature basis captures essential nonlinear oscillator
    physics:
    - Position: x, x*tanh(x), x^2 (spring force components)
    - Velocity: v, v*tanh(v), v^2 (damping force components)
    - Time: cos(t), sin(t) (periodic driving)
    - Cross: x*v (parametric coupling)
    - Constant offset
    
    All 10 parameters are fitted via BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to unit scale for better conditioning
    # t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    t_norm = (t - 40.0) / 10.0      # center ~40, scale ~10
    x_norm = x / 0.2                 # scale to ~unit
    v_norm = v / 0.35                # scale to ~unit
    
    # Position basis (spring force components)
    f_x1 = x_norm
    f_x2 = x_norm * np.tanh(x_norm)
    f_x3 = x_norm * x_norm
    
    # Velocity basis (damping force components)
    f_v1 = v_norm
    f_v2 = v_norm * np.tanh(v_norm)
    f_v3 = v_norm * v_norm
    
    # Time basis (periodic driving force)
    f_t1 = np.cos(t_norm)
    f_t2 = np.sin(t_norm)
    
    # Cross-coupling
    f_xv = x_norm * v_norm
    
    # Linear combination of features
    return (p0 * f_x1 + p1 * f_x2 + p2 * f_x3 +
            p3 * f_v1 + p4 * f_v2 + p5 * f_v3 +
            p6 * f_t1 + p7 * f_t2 +
            p8 * f_xv +
            p9)
