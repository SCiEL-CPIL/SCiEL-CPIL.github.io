import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Fourier basis on [-1, 1] with period 2 (fundamental frequency pi)
    # Orthogonal basis functions well-suited to the domain
    
    # First harmonics
    cos_x = np.cos(np.pi * x)
    sin_x = np.sin(np.pi * x)
    cos_y = np.cos(np.pi * y)
    sin_y = np.sin(np.pi * y)
    
    # Second harmonics
    cos2_x = np.cos(2 * np.pi * x)
    cos2_y = np.cos(2 * np.pi * y)
    
    # 10 basis functions:
    b0 = np.ones_like(x)           # p0: Constant
    b1 = cos_x                      # p1: cos(pi*x)
    b2 = cos_y                      # p2: cos(pi*y)
    b3 = sin_x                      # p3: sin(pi*x)
    b4 = sin_y                      # p4: sin(pi*y)
    b5 = cos2_x                     # p5: cos(2*pi*x)
    b6 = cos2_y                     # p6: cos(2*pi*y)
    b7 = cos_x * cos_y              # p7: cos(pi*x)*cos(pi*y)
    b8 = cos_x * sin_y              # p8: cos(pi*x)*sin(pi*y)
    b9 = sin_x * cos_y              # p9: sin(pi*x)*cos(pi*y)
    
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
