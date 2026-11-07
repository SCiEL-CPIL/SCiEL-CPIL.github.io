"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Start time to guard against potential infinite loops or slow operations
    # though Chebyshev evaluation is O(N), this is a safety measure per contract.
    t_start = time.time()
    
    # 10 parameters for a 9th-degree Chebyshev polynomial basis.
    # Chebyshev polynomials are orthogonal on [-1, 1] and provide better
    # numerical conditioning than monomials for approximating smooth functions.
    # We map the input domain [-2, 2] to [-1, 1] to utilize this property.
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t) using the recurrence relation:
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.empty((10, len(t)), dtype=np.float64)
    T[0, :] = 1.0
    if len(t) > 0:
        T[1, :] = t
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
            
    # The model is y = sum_{k=0}^{9} p_k * T_k(t)
    # This is a linear model in parameters, so BFGS will converge to the global minimum
    # of the MSE (which is a convex quadratic function).
    
    # Compute predictions: y = T @ params
    # T is (10, N), params is (10,) -> result is (N,)
    predictions = T @ params
    
    # Safety check for time (though this should be very fast)
    if time.time() - t_start > 110:
         # Return current best guess if we are running out of time
         # In this simple linear case, we don't have a "loop" to break out of,
         # but we ensure we don't hang.
         pass
        
    return predictions
# EVOLVE-BLOCK-END
