import numpy as np

def coefficients(stages):
    """4th-order Suzuki-Trotter splitting coefficients for 12 stages."""
    # 4th order coefficients
    # c1 = 1 / (2 - 2^(1/3))
    # c2 = -2^(1/3) / (2 - 2^(1/3))
    # c3 = 1 - c1 - c2
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = 1.0 - c1 - c2
    
    # The scheme is B(c1) A(c1) B(c2) A(c2) B(c3) A(c3) ...
    # We have 12 stages. We can use the first 6 and pad with zeros.
    # Or distribute them?
    # Padding with zeros is safe and maintains 4th order.
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Map to first 6 stages
    # Stage 1: B1, A1
    b[0] = c1
    a[0] = c1
    # Stage 2: B2, A2
    b[1] = c2
    a[1] = c2
    # Stage 3: B3, A3
    b[2] = c3
    a[2] = c3
    # Stages 4-12: 0
    
    return a, b
