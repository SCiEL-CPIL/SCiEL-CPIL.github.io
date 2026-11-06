import numpy as np

def coefficients(stages):
    # 6th-order Suzuki splitting coefficients for 7 factors
    # c = 2^(1/3)
    c = 2.0 ** (1.0 / 3.0)
    # 6th order 7-factor coefficients
    # A(α1) B(β1) A(α2) B(β2) A(α3) B(β3) A(α1)
    # α1 = 1 / (2 * (1 - c))
    # β1 = -α1 / c
    # α2 = 1 / (2 * (1 - c^2))
    # β2 = -α2 / c^2
    # α3 = 1 - 2*α1 - 2*α2
    # β3 = 1 - 2*β1 - 2*β2
    
    alpha1 = 1.0 / (2.0 * (1.0 - c))
    beta1 = -alpha1 / c
    alpha2 = 1.0 / (2.0 * (1.0 - c**2))
    beta2 = -alpha2 / (c**2)
    alpha3 = 1.0 - 2.0 * alpha1 - 2.0 * alpha2
    beta3 = 1.0 - 2.0 * beta1 - 2.0 * beta2
    
    # Logical sequence: A(α1) B(β1) A(α2) B(β2) A(α3) B(β3) A(α1)
    # 7 logical steps. Map to 12 stages.
    # Symmetric mapping:
    # Slots 1, 12: A(α1/2)
    # Slots 2, 11: B(β1/2)
    # Slots 3, 10: A(α2/2)
    # Slots 4, 9: B(β2/2)
    # Slots 5, 8: A(α3/2)
    # Slots 6, 7: B(β3/2)
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    a[0] = alpha1 / 2.0
    a[11] = alpha1 / 2.0
    
    b[1] = beta1 / 2.0
    b[10] = beta1 / 2.0
    
    a[2] = alpha2 / 2.0
    a[9] = alpha2 / 2.0
    
    b[3] = beta2 / 2.0
    b[8] = beta2 / 2.0
    
    a[4] = alpha3 / 2.0
    a[7] = alpha3 / 2.0
    
    b[5] = beta3 / 2.0
    b[6] = beta3 / 2.0
    
    # Normalize
    sa = sum(a)
    sb = sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    
    return a, b
