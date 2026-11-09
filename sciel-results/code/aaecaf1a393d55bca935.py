import numpy as np
import time
from scipy.optimize import minimize


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with bounded nonlinearities.
    
    Combines the structural idea of a parametric limit-cycle oscillator
    with a linear-in-parameters feature basis for stable optimization.
    
    Features:
    - Position: x, x*tanh(x), x^2 (spring force components)
    - Velocity: v, v*tanh(v), v^2 (damping force components)
    - Time: cos(t), sin(t) (periodic driving)
    - Cross: x*v (parametric coupling)
    - Constant offset
    
    All 10 parameters are fitted via BFGS starting from all ones.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Position basis (spring force components)
    f_x1 = x
    f_x2 = x * np.tanh(x)
    f_x3 = x * x
    
    # Velocity basis (damping force components)
    f_v1 = v
    f_v2 = v * np.tanh(v)
    f_v3 = v * v
    
    # Time basis (periodic driving force)
    f_t1 = np.cos(t)
    f_t2 = np.sin(t)
    
    # Cross-coupling
    f_xv = x * v
    
    # Linear combination of features
    return (p0 * f_x1 + p1 * f_x2 + p2 * f_x3 +
            p3 * f_v1 + p4 * f_v2 + p5 * f_v3 +
            p6 * f_t1 + p7 * f_t2 +
            p8 * f_xv +
            p9)
