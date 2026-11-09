"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis of degree 9.
    
    Strategy:
    1. Map input x from [-3, 3] to t in [-1, 1] via t = x / 3.
    2. Evaluate Chebyshev polynomials T_0(t) ... T_9(t).
    3. The output is the linear combination: y = sum_{k=0}^{9} c_k * T_k(t).
    
    This approach is structurally different from the previous composite model 
    (Gaussian + Sine + Poly) because it uses a single, well-conditioned basis 
    that is optimal for approximating smooth functions on a finite interval. 
    The linear nature of the parameters makes the optimization landscape 
    convex with respect to the coefficients (for a fixed basis), which 
    BFGS handles very efficiently, allowing it to converge to the global 
    minimum for the given basis expansion.
    """
    
    # Map x to [-1, 1] for Chebyshev polynomials
    # x is in [-3, 3], so t = x / 3
    t = x / 3.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    # T_6(t) = 32t^6 - 48t^4 + 18t^2 - 1
    # T_7(t) = 64t^7 - 112t^5 + 56t^3 - 7t
    # T_8(t) = 128t^8 - 256t^6 + 160t^4 - 32t^2 + 1
    # T_9(t) = 256t^9 - 576t^7 + 432t^5 - 120t^3 + 9t
    
    # We can compute these iteratively or use the closed forms.
    # Iterative is safer and faster for high degrees, but degree 9 is low.
    # Let's use the recurrence relation: T_{n+1}(t) = 2t T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # params[0] to params[9] are the coefficients c_0 to c_9
    # y = c_0 * T_0 + ... + c_9 * T_9
    
    # Efficiently compute the dot product
    # T is (10, N), params is (10,)
    y = T @ params
    
    # Ensure finite output
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
