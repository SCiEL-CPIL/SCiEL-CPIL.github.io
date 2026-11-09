"""
Symbolic regression using a Chebyshev polynomial basis (degree 9).
Chebyshev polynomials are orthogonal on [-1,1], providing well-conditioned
coefficients for BFGS optimization. The 10 parameters are the Chebyshev
coefficients, giving a degree-9 polynomial that can approximate smooth
functions on [-2,2] with near-uniform error distribution.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = sum_{k=0}^{9} p[k] * T_k((x)/2)
    
    where T_k is the k-th Chebyshev polynomial of the first kind,
    evaluated at t = x/2 (mapping [-2,2] to [-1,1]).
    
    Parameters:
        p[0..9]: Chebyshev coefficients
    
    This representation uses an orthogonal basis that is well-conditioned
    for gradient-based optimization, unlike monomial polynomials which
    suffer from ill-conditioning at higher degrees.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.ndim != 1:
        p = p.flatten()
    
    # Map x from [-2,2] to [-1,1] for Chebyshev domain
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1, T_1(t) = t, T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    result = p[0] * T0
    if len(p) > 1:
        result = result + p[1] * T1
    
    if len(p) > 2:
        T_prev2 = T0
        T_prev1 = T1
        for k in range(2, min(len(p), 10)):
            T_curr = 2.0 * t * T_prev1 - T_prev2
            result = result + p[k] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
    
    return result
# EVOLVE-BLOCK-END
