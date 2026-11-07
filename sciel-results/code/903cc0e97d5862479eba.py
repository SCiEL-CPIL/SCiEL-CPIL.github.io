"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    
    Structure:
    a = offset + drive + restoring(x) + damping(v) + coupling(x,v)
    
    Parameters:
    p[0]: offset
    p[1]: drive amplitude
    p[2]: drive frequency (normalized)
    p[3]: drive phase
    p[4]: linear restoring coeff
    p[5]: cubic restoring coeff
    p[6]: linear damping coeff
    p[7]: cubic damping coeff
    p[8]: cross term coeff (x*v)
    p[9]: quadratic restoring coeff (x^2)
    """
    # Normalize inputs to improve conditioning for BFGS
    # t in [30, 50] -> map to [-1, 1]
    t_norm = (t - 40.0) / 10.0
    
    # x in approx [-0.35, 0.35] -> map to approx [-1, 1]
    x_norm = x / 0.35
    
    # v in approx [-0.35, 0.35] -> map to approx [-1, 1]
    v_norm = v / 0.35
    
    # Components
    # Driving force: sinusoidal
    drive = params[1] * np.cos(params[2] * t_norm + params[3])
    
    # Restoring force: linear + quadratic + cubic (Duffing-like)
    restoring = (params[4] * x_norm + 
                 params[9] * x_norm**2 + 
                 params[5] * x_norm**3)
    
    # Damping force: linear + cubic
    damping = params[6] * v_norm + params[7] * v_norm**3
    
    # Coupling term
    coupling = params[8] * x_norm * v_norm
    
    # Total acceleration
    return params[0] + drive + restoring + damping + coupling
# EVOLVE-BLOCK-END
