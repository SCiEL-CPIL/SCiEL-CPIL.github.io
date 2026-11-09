import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1]
    # Center: 1.25, half-range: 0.75
    xn = (x - 1.25) / 0.75
    yn = (y - 1.25) / 0.75
    zn = (z - 1.25) / 0.75
    
    # Clip for safety
    xn = np.clip(xn, -1.0, 1.0)
    yn = np.clip(yn, -1.0, 1.0)
    zn = np.clip(zn, -1.0, 1.0)
    
    # Build a hybrid basis combining Chebyshev polynomials (for smooth polynomial trends)
    # with rational and trigonometric terms (for non-polynomial structure)
    # 
    # Chebyshev polynomials T_n(t):
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    
    # Chebyshev basis for each variable
    Tx1 = xn
    Tx2 = 2.0 * xn**2 - 1.0
    Ty1 = yn
    Ty2 = 2.0 * yn**2 - 1.0
    Tz1 = zn
    Tz2 = 2.0 * zn**2 - 1.0
    
    # 10 basis functions designed to capture diverse functional forms:
    # 0: constant
    # 1: T1(x) - linear x (Chebyshev)
    # 2: T1(y) - linear y (Chebyshev)
    # 3: T1(z) - linear z (Chebyshev)
    # 4: T2(x) - quadratic x (Chebyshev)
    # 5: T2(y) - quadratic y (Chebyshev)
    # 6: T2(z) - quadratic z (Chebyshev)
    # 7: xn * yn - interaction xy
    # 8: 1/(1+xn^2) + 1/(1+yn^2) + 1/(1+zn^2) - rational smoothing
    # 9: sin(pi*xn) * sin(pi*yn) * sin(pi*zn) - captures periodic/oscillatory structure
    
    b0 = np.ones_like(xn)
    b1 = Tx1
    b2 = Ty1
    b3 = Tz1
    b4 = Tx2
    b5 = Ty2
    b6 = Tz2
    b7 = xn * yn
    b8 = 1.0 / (1.0 + xn**2) + 1.0 / (1.0 + yn**2) + 1.0 / (1.0 + zn**2)
    b9 = np.sin(np.pi * xn) * np.sin(np.pi * yn) * np.sin(np.pi * zn)
    
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we've exceeded a safety margin, return immediately
    if time.time() - start_time > 120.0:
        return result
    
    return result
