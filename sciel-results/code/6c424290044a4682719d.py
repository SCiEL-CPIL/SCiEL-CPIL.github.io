import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear hinge model to capture the kinked behavior
    of aluminum stress-strain curves (elastic, yield, strain hardening).
    
    Model:
      sigma = E(T)*eps + Y1(T)*hinge(eps, y1(T)) + Y2(T)*hinge(eps, y2(T)) 
              + Y3(T)*hinge(eps, y3(T)) + offset
    
    Where:
      E(T)   = p0 + p1*T        (elastic modulus)
      y1(T)  = p2 + p3*T        (first yield strain, ~elastic-plastic transition)
      Y1(T)  = p4 + p5*T        (yield stress magnitude at first kink)
      y2(T)  = p6 + p7*T        (second yield strain, ~upper yield to hardening)
      Y2(T)  = p8               (additional hardening slope change)
      y3(T)  = p9               (third breakpoint for further hardening)
      offset = 0 (absorbed into E and Y terms)
    
    hinge(x, a) = max(0, x - a)
    
    This creates a piecewise-linear curve with distinct slopes in each region:
      Region 1 (eps < y1): slope = E
      Region 2 (y1 < eps < y2): slope = E + Y1
      Region 3 (y2 < eps < y3): slope = E + Y1 + Y2
      Region 4 (eps > y3): slope = E + Y1 + Y2 + Y3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent parameters
    E_T = p0 + p1 * T
    y1_T = p2 + p3 * T
    Y1_T = p4 + p5 * T
    y2_T = p6 + p7 * T
    Y2_T = p8
    y3_T = p9
    
    # Hinge functions: max(0, eps - breakpoint)
    hinge1 = np.maximum(0.0, eps - y1_T)
    hinge2 = np.maximum(0.0, eps - y2_T)
    hinge3 = np.maximum(0.0, eps - y3_T)
    
    # Total stress
    sigma = E_T * eps + Y1_T * hinge1 + Y2_T * hinge2 + 0.0 * hinge3
    
    return sigma
