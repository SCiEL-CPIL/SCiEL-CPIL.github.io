"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 parameters.
    Combines Chebyshev polynomials (global smooth trend) with RBFs (local features)
    and a log term (captures potential singular behavior at x=0).
    This structure is linear in params, allowing BFGS to converge efficiently.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)

    # Scale x from [0.1, 6] to [-1, 1] for better numerical conditioning
    # Midpoint: 3.05, Half-range: 2.95
    t = (x - 3.05) / 2.95
    
    # Check time
    if time.time() > deadline:
        return np.zeros_like(x, dtype=np.float64)

    try:
        # Basis functions:
        # 1-5: Chebyshev polynomials T_0 to T_4 (well-conditioned global basis)
        phi0 = 1.0
        phi1 = t
        phi2 = 2.0 * t**2 - 1.0
        phi3 = 4.0 * t**3 - 3.0 * t
        phi4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
        
        # 6-9: RBFs (Gaussian kernels) centered at -0.75, -0.25, 0.25, 0.75
        # Width chosen to be narrow enough for local features but wide enough
        # to avoid numerical issues with BFGS. lambda=10.0
        lam = 10.0
        centers = [-0.75, -0.25, 0.25, 0.75]
        
        phi5 = np.exp(-lam * (t - centers[0])**2)
        phi6 = np.exp(-lam * (t - centers[1])**2)
        phi7 = np.exp(-lam * (t - centers[2])**2)
        phi8 = np.exp(-lam * (t - centers[3])**2)
        
        # 10: Log term (captures 1/x or log(x) behavior, safe for x in [0.1, 6])
        phi9 = np.log(x)
        
        result = (p[0] * phi0 +
                  p[1] * phi1 +
                  p[2] * phi2 +
                  p[3] * phi3 +
                  p[4] * phi4 +
                  p[5] * phi5 +
                  p[6] * phi6 +
                  p[7] * phi7 +
                  p[8] * phi8 +
                  p[9] * phi9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
