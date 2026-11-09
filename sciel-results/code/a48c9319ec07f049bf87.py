import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Tensor-product Fourier basis with sine and cosine terms
    # Using frequencies 1 and 2 for both x and y to capture smooth periodic interactions
    # This provides an orthogonal basis that is numerically stable for BFGS optimization
    
    # Cosine terms for x
    cos_x1 = np.cos(np.pi * x)
    cos_x2 = np.cos(2 * np.pi * x)
    
    # Sine terms for x
    sin_x1 = np.sin(np.pi * x)
    sin_x2 = np.sin(2 * np.pi * x)
    
    # Cosine terms for y
    cos_y1 = np.cos(np.pi * y)
    cos_y2 = np.cos(2 * np.pi * y)
    
    # Sine terms for y
    sin_y1 = np.sin(np.pi * y)
    sin_y2 = np.sin(2 * np.pi * y)
    
    # Construct 10 basis terms: constant, x-terms, y-terms, and interactions
    b0 = np.ones_like(x)  # constant
    b1 = cos_x1           # cos(πx)
    b2 = sin_x1           # sin(πx)
    b3 = cos_y1           # cos(πy)
    b4 = sin_y1           # sin(πy)
    b5 = cos_x1 * cos_y1  # interaction cos(πx)cos(πy)
    b6 = sin_x1 * sin_y1  # interaction sin(πx)sin(πy)
    b7 = cos_x2           # cos(2πx)
    b8 = cos_y2           # cos(2πy)
    b9 = sin_x1 * cos_y1  # cross term sin(πx)cos(πy)
    
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
