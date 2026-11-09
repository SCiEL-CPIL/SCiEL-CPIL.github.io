import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a mixture of 3 adaptive Gaussian kernels 
    plus a global bias.
    
    Parameters (length 10):
    p0: Global bias
    p1: Amplitude of Gaussian 1
    p2: Center of Gaussian 1
    p3: Log-Width of Gaussian 1 (width = exp(p3))
    p4: Amplitude of Gaussian 2
    p5: Center offset of Gaussian 2 (center = p5 + 1.0)
    p6: Log-Width of Gaussian 2 (width = exp(p6))
    p7: Amplitude of Gaussian 3
    p8: Center offset of Gaussian 3 (center = p8 - 1.0)
    p9: Log-Width of Gaussian 3 (width = exp(p9))
    
    The fixed offsets (+1.0 and -1.0) break symmetry at initialization (all ones),
    placing initial centers at x=1, x=2, and x=0, allowing the optimizer to
    explore distinct local features.
    """
    x = np.asarray(x)
    
    # Unpack parameters
    p = params
    b = p[0]
    
    # Gaussian 1
    a1 = p[1]
    c1 = p[2]
    w1 = np.exp(p[3])
    # Avoid division by zero if w1 is tiny, though exp is always > 0.
    # If p[3] is very negative, w1 is small, term is sharp.
    g1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    
    # Gaussian 2 (Center shifted by +1.0 to break symmetry)
    a2 = p[4]
    c2 = p[5] + 1.0
    w2 = np.exp(p[6])
    g2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    
    # Gaussian 3 (Center shifted by -1.0 to break symmetry)
    a3 = p[7]
    c3 = p[8] - 1.0
    w3 = np.exp(p[9])
    g3 = a3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    y = b + g1 + g2 + g3
    
    return y
