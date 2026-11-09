import numpy as np
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (global) + 5 Gaussian RBFs (local).
    params[0:5] -> coefficients for Chebyshev T_0 to T_4
    params[5:10] -> coefficients for Gaussian RBFs at fixed centers
    """
    x_safe = np.asarray(x, dtype=np.float64)
    if x_safe.ndim == 0:
        x_safe = x_safe.reshape(1)
    
    # 1. Chebyshev Polynomials (Degrees 0-4)
    # Map x from [0.1, 6] to [-1, 1] for standard Chebyshev stability
    # t = (2*x - (max+min)) / (max-min)
    # min=0.1, max=6.0
    # t = (2*x - 6.1) / 5.9
    t = (2.0 * x_safe - 6.1) / 5.9
    
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # 2. Gaussian Radial Basis Functions
    # Fixed centers distributed across [0.1, 6]
    # Centers: 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 (Using 5 of them, or spread evenly)
    # Let's use centers at 0.5, 2.0, 3.5, 5.0, 5.8 to cover the range well
    # Bandwidths (sigma) chosen to be broad enough to overlap but distinct
    # We use fixed sigmas to keep the problem linear in parameters.
    # If sigmas were parameters, it would be non-convex.
    
    centers = np.array([0.5, 2.0, 3.5, 5.0, 5.8])
    sigmas = np.array([1.2, 1.5, 1.5, 1.2, 0.8])
    
    # Compute RBFs: exp(-((x - c)^2) / (2 * sigma^2))
    # Vectorized computation
    # x_safe shape: (N,), centers shape: (5,)
    # diff shape: (N, 5)
    diff = x_safe[:, np.newaxis] - centers[np.newaxis, :]
    
    # Avoid overflow in exp by clipping the exponent
    # exponent = -diff^2 / (2 * sigma^2)
    inv_var = 1.0 / (2.0 * sigmas**2)
    exponent = -diff**2 * inv_var[np.newaxis, :]
    
    # Clip exponent to prevent underflow/overflow issues, though exp(-large) is 0
    # exp(-100) is ~ 3e-44, which is fine. exp(100) would be bad, but diff^2 is positive.
    RBFs = np.exp(exponent)
    
    # 3. Linear Combination
    # pred = sum(params[0:5] * T) + sum(params[5:10] * RBFs)
    
    # T terms
    pred = (params[0] * T0 + 
            params[1] * T1 + 
            params[2] * T2 + 
            params[3] * T3 + 
            params[4] * T4)
    
    # RBF terms
    # RBFs is (N, 5), params[5:10] is (5,)
    # Dot product: (N, 5) @ (5,) -> (N,)
    pred += RBFs @ params[5:10]
    
    return pred
