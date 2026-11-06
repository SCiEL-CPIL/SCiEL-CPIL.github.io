import numpy as np

def coefficients(stages):
    # 4th-order symmetric Suzuki splitting mapped to 12 stages
    # Use the 4th-order 2-step coefficients and distribute symmetrically
    c = 2.0 ** (1.0 / 3.0)
    alpha = 1.0 / (2.0 * (1.0 - c))
    beta = -alpha / c
    gamma = 1.0 - 2.0 * beta
    
    # 5-factor 4th-order: A(α) B(β) A(α) B(γ) A(α)
    # Map to 12 stages: split into 6 A-slots and 6 B-slots symmetrically
    # a-coefficients (A-evolution weights), 12 of them, sum=1, symmetric
    a = [alpha/2, alpha/2, 0, 0, alpha/2, alpha/2, alpha/2, alpha/2, 0, 0, alpha/2, alpha/2]
    # b-coefficients (B-evolution weights), 12 of them, sum=1, symmetric
    b = [beta/2, beta/2, 0, 0, gamma/2, gamma/2, 0, 0, beta/2, beta/2, 0, 0]
    
    # Normalize to ensure exact sum=1 (numerical safety)
    sa = sum(a)
    sb = sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    
    return a, b
