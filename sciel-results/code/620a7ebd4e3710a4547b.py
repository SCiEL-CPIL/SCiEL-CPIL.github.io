"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Use Chebyshev polynomials T0-T9 as a well-conditioned basis that spans
polynomial space up to degree 9. The model is linear in parameters, so BFGS
converges to the global least-squares solution efficiently.

Chebyshev polynomials are optimal for uniform approximation on [-1,1] and
provide a nearly orthogonal basis that minimizes condition number issues.
A degree-9 Chebyshev series can approximate any sufficiently smooth function
on [-3,3] (after scaling) with high fidelity.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with 10 parameters using Chebyshev polynomials T0-T9.
    
    Basis functions (all in scaled coordinate t = x/3 in [-1,1]):
    0: T0(t) = 1
    1: T1(t) = t
    2: T2(t) = 2t^2 - 1
    3: T3(t) = 4t^3 - 3t
    4: T4(t) = 8t^4 - 8t^2 + 1
    5: T5(t) = 16t^5 - 20t^3 + 5t
    6: T6(t) = 32t^6 - 48t^4 + 18t^2 - 1
    7: T7(t) = 64t^7 - 112t^5 + 56t^3 - 7t
    8: T8(t) = 128t^8 - 256t^6 + 160t^4 - 32t^2 + 1
    9: T9(t) = 256t^9 - 576t^7 + 432t^5 - 120t^3 + 9t
    
    The model is linear in params, ensuring convex loss and reliable BFGS convergence.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is at least 1-D
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x_flat / 3.0
    
    # Build Chebyshev polynomials T0 through T9 using recurrence
    # T_{n+1}(t) = 2t * T_n(t) - T_{n-1}(t)
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Compute T2 through T9 via recurrence for numerical stability
    T = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * T[n-1] - T[n-2]
        T.append(Tn)
    
    # Combine: result = sum_{n=0}^{9} p[n] * Tn(t)
    result = np.zeros_like(t)
    for n in range(10):
        result += p[n] * T[n]
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
