import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod with temperature dependence.
    
    Uses a smooth bilinear elastic-plastic model with:
    - Temperature-dependent Young's modulus
    - Temperature-dependent yield strain
    - Smooth transition via tanh
    - Plastic hardening (linear + quadratic)
    - Temperature-dependent yield stress
    """
    s = strain
    t = temp
    p = params
    
    # Extract parameters with safety clipping to avoid overflow
    p0 = p[0]  # Base Young's modulus
    p1 = p[1]  # Modulus temp coefficient
    p2 = p[2]  # Base yield strain
    p3 = p[3]  # Yield strain temp coefficient
    p4 = p[4]  # Transition sharpness
    p5 = p[5]  # Base yield stress
    p6 = p[6]  # Yield stress temp coefficient
    p7 = p[7]  # Linear hardening
    p8 = p[8]  # Quadratic hardening
    p9 = p[9]  # Offset
    
    # Temperature-dependent modulus
    E = p0 + p1 * t
    
    # Temperature-dependent yield strain
    sy = p2 + p3 * t
    
    # Smooth transition function (0 = elastic, 1 = plastic)
    # Clip argument to prevent tanh overflow issues
    arg = np.clip(p4 * (s - sy), -50, 50)
    tr = 0.5 * (1.0 + np.tanh(arg))
    
    # Elastic branch
    el = E * s * (1.0 - tr)
    
    # Plastic branch: yield stress + hardening
    y0 = p5 + p6 * t
    hard = p7 * s + p8 * s * s
    pl = y0 + hard
    
    # Combine
    result = el + pl * tr + p9
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
