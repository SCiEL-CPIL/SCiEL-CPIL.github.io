import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a dense Chebyshev polynomial basis.
    
    Strategy:
    - Uses the 10 available parameters as coefficients for a degree-9 Chebyshev polynomial.
    - Chebyshev polynomials are orthogonal on [-1, 1] (mapped from [-2, 2]), providing
      superior conditioning and approximation properties compared to monomials for
      smooth functions on a bounded interval.
    - This is a global, fixed-basis approach, structurally different from the local
      adaptive RBF approach in the parent. It relies on the global smoothness of the
      target function and the efficiency of spectral methods for such problems.
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure x is 1D
    x = np.asarray(x).flatten()
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t) using the recurrence relation:
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    # Precompute basis vectors for efficiency
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Build the basis matrix B where B[:, i] is T_i(t)
    # We need T_0 to T_9 (10 basis functions for 10 params)
    basis = np.empty((len(t), 10), dtype=np.float64)
    basis[:, 0] = T0
    basis[:, 1] = T1
    
    T_prev = T0
    T_curr = T1
    
    for i in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        basis[:, i] = T_next
        T_prev = T_curr
        T_curr = T_next
        
        # Optional: early exit if time is running out, though this is very fast
        if time.time() - start_time > 110.0:
             # Fill remaining with zeros or break, but 10 steps is trivial
             break

    # The model is sum(params[i] * T_i(t))
    # params is length 10.
    # We use all 10 parameters directly as coefficients.
    
    # Linear combination
    result = basis @ params
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
