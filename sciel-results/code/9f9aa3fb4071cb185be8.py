import numpy as np

def coefficients(stages):
    """
    Uses a 4th order Yoshida splitting with 4 substeps of 3 stages each.
    Total 12 stages.
    """
    # Yoshida 4th order 3-stage (3 A, 3 B) coefficients
    # gamma = 1 / (2 - 2**(1/4))
    gamma = 1.0 / (2.0 - 2.0 ** 0.25)
    
    x1 = 1.0 / (2.0 * (1.0 - gamma))
    x2 = -gamma / (2.0 * (1.0 - gamma))
    x3 = 0.5
    
    # We have 12 stages. We use 4 substeps.
    # Each substep has 3 A's and 3 B's.
    # Coefficients for each substep are scaled by 1/4.
    
    a_sub = [x1 / 4.0, x2 / 4.0, x3 / 4.0]
    b_sub = [x1 / 4.0, x2 / 4.0, x3 / 4.0]
    
    a = []
    b = []
    for _ in range(4):
        a.extend(a_sub)
        b.extend(b_sub)
        
    # Ensure length is exactly stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    # Truncate if necessary (should not happen for stages=12)
    a = a[:stages]
    b = b[:stages]
    
    return a, b
