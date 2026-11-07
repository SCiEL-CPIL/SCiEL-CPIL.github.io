# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        # Pad with zeros
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    
    # Strategy: Linear combination of 10 diverse basis functions
    # This transforms the problem into a linear regression in parameter space,
    # which BFGS can solve extremely quickly and robustly (quadratic loss surface).
    # The basis functions are chosen to cover various shapes:
    # 1. Constant (1)
    # 2. Linear (x)
    # 3. Quadratic (x^2)
    # 4. Cubic (x^3)
    # 5. Quartic (x^4)
    # 6. Sine (sin(x))
    # 7. Cosine (cos(x))
    # 8. Sine 2x (sin(2x))
    # 9. Cosine 2x (cos(2x))
    # 10. Radial Basis Function centered at 0 (exp(-x^2))
    
    # Precompute basis functions for efficiency
    # x is shape (N,)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    b9 = np.exp(-x ** 2)
    
    # Combine linearly
    # result = p[0]*b0 + p[1]*b1 + ... + p[9]*b9
    
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
