import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Structural change: use interaction/multiplicative basis to capture
    # non-additive structure common in scientific functions
    # Basis: {1, x, y, z, x*y, x*z, y*z, x*y*z, x/z, y/z}
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = x * y
    b5 = x * z
    b6 = y * z
    b7 = x * y * z
    b8 = x / z
    b9 = y / z
    
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
