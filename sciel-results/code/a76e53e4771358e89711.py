import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Fourier basis on [-1, 1] with period 2 (fundamental frequency pi)
    # This basis is orthogonal on the domain and well-suited for smooth functions.
    
    # Harmonics for x
    c1x = np.cos(np.pi * x)
    s1x = np.sin(np.pi * x)
    c2x = np.cos(2 * np.pi * x)
    s2x = np.sin(2 * np.pi * x)
    
    # Harmonics for y
    c1y = np.cos(np.pi * y)
    s1y = np.sin(np.pi * y)
    c2y = np.cos(2 * np.pi * y)
    s2y = np.sin(2 * np.pi * y)
    
    # Basis functions
    # p0: Constant
    b0 = np.ones_like(x)
    # p1-p2: First harmonic x
    b1 = c1x
    b2 = s1x
    # p3-p4: First harmonic y
    b3 = c1y
    b4 = s1y
    # p5-p6: Second harmonic x
    b5 = c2x
    b6 = s2x
    # p7-p8: Second harmonic y
    b7 = c2y
    b8 = s2y
    # p9: First order interaction (cos x * cos y)
    b9 = c1x * c1y
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
