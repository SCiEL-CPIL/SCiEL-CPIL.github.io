import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to t in [0, 1]
    t = (x - 0.1) / 5.9
    
    y = np.zeros_like(x)
    
    # p0: constant
    y += params[0] * 1.0
    
    # p1, p2: freq 1
    y += params[1] * np.sqrt(2.0) * np.cos(2.0 * np.pi * t)
    y += params[2] * np.sqrt(2.0) * np.sin(2.0 * np.pi * t)
    
    # p3, p4: freq 2
    y += params[3] * np.sqrt(2.0) * np.cos(4.0 * np.pi * t)
    y += params[4] * np.sqrt(2.0) * np.sin(4.0 * np.pi * t)
    
    # p5, p6: freq 3
    y += params[5] * np.sqrt(2.0) * np.cos(6.0 * np.pi * t)
    y += params[6] * np.sqrt(2.0) * np.sin(6.0 * np.pi * t)
    
    # p7, p8: freq 4
    y += params[7] * np.sqrt(2.0) * np.cos(8.0 * np.pi * t)
    y += params[8] * np.sqrt(2.0) * np.sin(8.0 * np.pi * t)
    
    # p9: freq 5 (cosine only to fill 10 params, or maybe linear?)
    # Let's stick to Fourier for consistency, but maybe a linear term is safer?
    # If we use a linear term, it's not orthogonal to the constant.
    # Let's use another cosine.
    y += params[9] * np.sqrt(2.0) * np.cos(10.0 * np.pi * t)
    
    return y
