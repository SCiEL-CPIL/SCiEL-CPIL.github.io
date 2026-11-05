import numpy as np

def coefficients(stages):
    """
    4th-order Suzuki-Trotter splitting repeated twice (2 substeps).
    Uses 10 non-zero stages out of 12, with 2 zero stages.
    """
    c = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - c)
    w2 = -c / (2.0 - c)
    w3 = 1.0 - 2.0 * w1
    
    # S4(t/2) coefficients for B and A blocks
    # S4(t) = e^{t/2 B} e^{t/2 A} e^{w2 t B} e^{w2 t A} e^{w3 t B} e^{w3 t A} e^{w2 t B} e^{w2 t A} e^{t/2 B} e^{t/2 A}
    # For t/2:
    b_half = [0.25, w2 / 2.0, w3 / 2.0, w2 / 2.0, 0.25]
    a_half = [0.25, w2 / 2.0, w3 / 2.0, w2 / 2.0, 0.25]
    
    # Repeat twice
    a = a_half + a_half
    b = b_half + b_half
    
    # Pad to 12 stages with zeros
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    # Truncate if necessary (though stages is fixed at 12)
    a = a[:stages]
    b = b[:stages]
    
    return a, b
