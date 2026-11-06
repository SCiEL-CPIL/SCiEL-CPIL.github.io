import numpy as np

def coefficients(stages):
    """
    4th-order Yoshida product formula repeated 4 times to form a 12-stage scheme.
    """
    if stages != 12:
        # Fallback to Strang if stages != 12, though task specifies 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th order Yoshida coefficients for 3-stage scheme
    # c1 = c3 = 1 / (2 - 2^(1/3))
    # c2 = 1 - 2*c1
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = 1.0 - 2.0 * c1
    c3 = c1
    
    # d1 = d2 = 0.5
    d1 = 0.5
    d2 = 0.5
    
    # Repeat 4 times to get 12 stages
    a = []
    b = []
    for _ in range(4):
        a.extend([c1/4.0, c2/4.0, c3/4.0])
        b.extend([d1/4.0, d2/4.0, 0.0])
        
    # Ensure lengths are 12
    # a has 12 elements
    # b has 12 elements (8 non-zero, 4 zero)
    
    return a, b
