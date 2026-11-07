import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x_flat = np.asarray(x).flatten()
    p = np.asarray(params).flatten()
    
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    
    w1 = np.clip(p[4], -10.0, 10.0)
    c_sin1 = p[5]
    c_cos1 = p[6]
    
    w2 = np.clip(p[7], -10.0, 10.0)
    c_sin2 = p[8]
    c_cos2 = p[9]
    
    # Compute basis
    # Polynomial part
    poly_part = c0 + c1 * x_flat + c2 * x_flat**2 + c3 * x_flat**3
    
    # Trig part
    sin1 = np.sin(w1 * x_flat)
    cos1 = np.cos(w1 * x_flat)
    sin2 = np.sin(w2 * x_flat)
    cos2 = np.cos(w2 * x_flat)
    
    trig_part = c_sin1 * sin1 + c_cos1 * cos1 + c_sin2 * sin2 + c_cos2 * cos2
    
    result = poly_part + trig_part
    
    if not np.all(np.isfinite(result)):
        return np.full_like(x_flat, 1e10)
        
    return result
