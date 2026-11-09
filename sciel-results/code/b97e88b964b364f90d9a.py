import numpy as np
import time

# Precompute Chebyshev nodes and basis functions on the normalized domain
# Domain: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structured Chebyshev polynomial expansion with data-dependent normalization.
    """
    start_time = time.time()
    
    # Normalize inputs to [-1, 1]
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Compute Chebyshev basis functions
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    # x-basis
    x0 = np.ones_like(xn)
    x1 = xn
    x2 = 2 * xn**2 - 1
    x3 = 4 * xn**3 - 3 * xn
    
    # v-basis
    v0 = np.ones_like(vn)
    v1 = vn
    v2 = 2 * vn**2 - 1
    v3 = 4 * vn**3 - 3 * vn
    
    # Construct the 10 terms
    # 0: T0(x)T0(v)
    # 1: T1(x)T0(v)
    # 2: T0(x)T1(v)
    # 3: T2(x)T0(v)
    # 4: T0(x)T2(v)
    # 5: T1(x)T1(v)
    # 6: T3(x)T0(v)
    # 7: T0(x)T3(v)
    # 8: T2(x)T1(v)
    # 9: T1(x)T2(v)
    
    term0 = x0 * v0
    term1 = x1 * v0
    term2 = x0 * v1
    term3 = x2 * v0
    term4 = x0 * v2
    term5 = x1 * v1
    term6 = x3 * v0
    term7 = x0 * v3
    term8 = x2 * v1
    term9 = x1 * v2
    
    # Stack into matrix B (N, 10)
    B = np.column_stack((term0, term1, term2, term3, term4, term5, term6, term7, term8, term9))
    
    # Data-dependent normalization to improve conditioning
    # Compute norms of each column
    norms = np.linalg.norm(B, axis=0)
    # Avoid division by zero
    norms = np.where(norms < 1e-8, 1.0, norms)
    B_norm = B / norms
    
    # Combine with parameters
    # params are coefficients for the normalized basis
    result = B_norm @ params
    
    # Time guard
    if time.time() - start_time > 110:
         # Return a simple fallback if taking too long (unlikely)
         return np.zeros_like(x)
        
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
