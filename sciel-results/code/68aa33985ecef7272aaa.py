import numpy as np
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
        
    # Unpack parameters
    A1 = p[0]
    f1 = p[1]
    ph1 = p[2]
    
    A2 = p[3]
    c2 = p[4]
    w2 = p[5]
    
    A3 = p[6]
    c3 = p[7]
    s3 = p[8]
    
    C = p[9]
    
    # Stability measures
    # Ensure width is not zero
    w2_safe = w2 * w2 + 1e-6
    # Ensure frequency is not zero (optional, sin(0) is fine)
    
    # Component 1: Sinusoid
    y1 = A1 * np.sin(f1 * x + ph1)
    
    # Component 2: Gaussian
    # Use w2^2 in denominator to ensure positive variance
    y2 = A2 * np.exp(-((x - c2) ** 2) / (2.0 * w2_safe))
    
    # Component 3: Tanh
    y3 = A3 * np.tanh(s3 * x + c3)
    
    # Constant
    y4 = C
    
    result = y1 + y2 + y3 + y4
    
    # Clip result to prevent overflow in MSE calculation if params explode
    result = np.clip(result, -1e6, 1e6)
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
