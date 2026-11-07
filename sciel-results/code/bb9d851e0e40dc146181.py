"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Use Chebyshev polynomials T0-T9 on scaled x in [-1,1] for optimal
polynomial approximation. The model is linear in parameters, so BFGS converges
to the global least-squares solution efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model with 10 Chebyshev polynomial basis functions T0(x)-T9(x)
    where x is scaled from [-3,3] to [-1,1].
    
    Chebyshev polynomials minimize the maximum error of polynomial approximation
    on [-1,1], making them ideal for fitting smooth functions on a bounded domain.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Scale x from [-3, 3] to [-1, 1]
    t = x_flat / 3.0
    
    # Compute Chebyshev polynomials T0 through T9 using recurrence
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    # Combine linearly
    result = (p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4 +
              p[5] * T5 + p[6] * T6 + p[7] * T7 + p[8] * T8 + p[9] * T9)
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
