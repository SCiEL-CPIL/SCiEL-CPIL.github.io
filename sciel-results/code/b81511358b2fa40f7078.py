import numpy as np

def coefficients(stages):
    x = 1.0 / (2.0 - 2.0**(1.0/3.0))
    alpha1 = 1.0 / (2.0 * (1.0 - x))
    alpha2 = 0.5 - alpha1
    beta1 = 1.0 / (2.0 * x)
    beta2 = (1.0 - 2.0 * x) / x
    
    scale = 1.0 / 3.0
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    if stages >= 10:
        a[0] = scale * alpha1
        b[1] = scale * beta1; a[1] = scale * alpha2
        b[2] = scale * beta2; a[2] = scale * alpha2
        b[3] = scale * beta1; a[3] = scale * 2 * alpha1
        
        b[4] = scale * beta1; a[4] = scale * alpha2
        b[5] = scale * beta2; a[5] = scale * alpha2
        b[6] = scale * beta1; a[6] = scale * 2 * alpha1
        
        b[7] = scale * beta1; a[7] = scale * alpha2
        b[8] = scale * beta2; a[8] = scale * alpha2
        b[9] = scale * beta1; a[9] = scale * alpha1
        
    return a, b
