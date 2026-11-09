import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear hinge model to capture the kinked behavior
    of aluminum stress-strain curves (elastic, yield, strain hardening).
    
    Model:
      sigma = E(T)*eps + Y(T)*hinge(eps, y1(T)) + H(T)*hinge(eps, y2(T)) + offset
    
    Where:
      E(T)  = p0 + p1*T         (elastic modulus)
      y1(T) = p2 + p3*T         (first yield strain, ~elastic-plastic transition)
      Y(T)  = p4                (yield stress magnitude)
      y2(T) = p5 + p6*T         (second yield strain, ~upper-to-lower yield)
      H(T)  = p7 + p8*T         (hardening modulus)
      offset = p9               (intercept / offset)
    
    hinge(x, c) = max(0, x - c)
    
    This gives a three-segment piecewise-linear response:
      - Below y1: purely elastic (slope E)
      - Between y1 and y2: elastic + yield plateau (slope E + Y)
      - Above y2: elastic + yield + hardening (slope E + Y + H)
    
    The hinge functions create natural kinks at the yield points,
    which is more physically accurate for metals than smooth models.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent parameters
    E_T = p0 + p1 * T
    y1_T = p2 + p3 * T
    Y_T = p4
    y2_T = p5 + p6 * T
    H_T = p7 + p8 * T
    offset = p9
    
    # Hinge functions: max(0, eps - yield_strain)
    hinge1 = np.maximum(0.0, eps - y1_T)
    hinge2 = np.maximum(0.0, eps - y2_T)
    
    # Total stress: elastic + yield contribution + hardening contribution + offset
    sigma = E_T * eps + Y_T * hinge1 + H_T * hinge2 + offset
    
    return sigma
