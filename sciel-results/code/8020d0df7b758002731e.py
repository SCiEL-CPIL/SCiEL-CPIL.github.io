import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Tensor-product Fourier basis with 10 parameters
    # This models global periodicity and smooth interactions
    # Basis functions:
    # 0: 1 (bias)
    # 1: sin(pi*x)
    # 2: cos(pi*x)
    # 3: sin(pi*y)
    # 4: cos(pi*y)
    # 5: sin(pi*x)*sin(pi*y)
    # 6: sin(pi*x)*cos(pi*y)
    # 7: cos(pi*x)*sin(pi*y)
    # 8: cos(pi*x)*cos(pi*y)
    # 9: sin(2*pi*x)*sin(2*pi*y) (second harmonic interaction)
    
    p = params
    
    # Precompute trigonometric terms
    sin_px = np.sin(np.pi * x)
    cos_px = np.cos(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_py = np.cos(np.pi * y)
    sin_2px = np.sin(2 * np.pi * x)
    sin_2py = np.sin(2 * np.pi * y)
    
    # Build prediction
    pred = (p[0] + 
            p[1] * sin_px + 
            p[2] * cos_px + 
            p[3] * sin_py + 
            p[4] * cos_py + 
            p[5] * sin_px * sin_py + 
            p[6] * sin_px * cos_py + 
            p[7] * cos_px * sin_py + 
            p[8] * cos_px * cos_py + 
            p[9] * sin_2px * sin_2py)
    
    return pred
