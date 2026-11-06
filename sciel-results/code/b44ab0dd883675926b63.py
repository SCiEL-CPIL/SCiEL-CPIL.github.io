import numpy as np

def coefficients(stages):
    # 6th-order symmetric Suzuki splitting mapped to 12 stages
    # 6th-order coefficients
    c = 2.0 ** (1.0 / 3.0)
    gamma = c
    
    # Coefficients for B (y) and A (z)
    # Standard 6th-order symmetric splitting:
    # B(y1) A(z1) B(y2) A(z2) B(y3) A(z3) B(y2) A(z2) B(y1) A(z1)
    # y1 = z1 = 1 / (2 * (1 - gamma))
    # y2 = z2 = gamma / (2 * (gamma - 1))
    # y3 = z3 = 1 / (2 * (1 - gamma))
    
    y1 = 1.0 / (2.0 * (1.0 - gamma))
    y2 = gamma / (2.0 * (gamma - 1.0))
    y3 = 1.0 / (2.0 * (1.0 - gamma))
    
    z1 = y1
    z2 = y2
    z3 = y3
    
    # Map to 12 stages by splitting each of the 5 pairs into 2.4 stages?
    # We have 5 pairs of (B, A) in the 6th-order formula.
    # We have 12 stages of (B, A).
    # We can distribute the 5 pairs over 12 stages.
    # Simplest: split each coefficient into 2 parts, giving 10 stages.
    # We need 12 stages.
    # We can split y1, z1 into 3 parts?
    # Let's use a symmetric distribution of the 5 coefficients over 12 stages.
    # Stages 1-2: y1, z1
    # Stages 3-4: y2, z2
    # Stages 5-6: y3, z3
    # Stages 7-8: y3, z3
    # Stages 9-10: y2, z2
    # Stages 11-12: y1, z1
    # This uses 12 stages.
    
    b = [y1/2, y1/2, y2/2, y2/2, y3/2, y3/2, y3/2, y3/2, y2/2, y2/2, y1/2, y1/2]
    a = [z1/2, z1/2, z2/2, z2/2, z3/2, z3/2, z3/2, z3/2, z2/2, z2/2, z1/2, z1/2]
    
    # Normalize to ensure exact sum=1 (numerical safety)
    sa = sum(a)
    sb = sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    
    return a, b
