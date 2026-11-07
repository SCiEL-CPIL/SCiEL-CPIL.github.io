"""
Strategy: Chebyshev Polynomial Basis Model.
The model is:
f(x) = sum_{i=0}^{9} c_i * T_i(x/2)

Where T_i are Chebyshev polynomials of the first kind.
We map x from [-2, 2] to t in [-1, 1] via t = x/2.

Parameters (length 10):
params[0:10] : Coefficients c_0 to c_9 for the Chebyshev basis.

This approach is linear in the parameters, making the optimization problem convex.
BFGS will reliably find the global minimum for the given basis.
Chebyshev polynomials are orthogonal on [-1,1] and provide a complete,
well-conditioned representation for smooth functions, capturing structure
much better than a low-degree spline with the same number of parameters.
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    try:
        # Map x from [-2, 2] to t in [-1, 1]
        t = x / 2.0
        
        # Compute Chebyshev polynomials T_0 through T_9 using recurrence
        # T_0(t) = 1, T_1(t) = t, T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
        T = np.zeros((10, x.shape[0]))
        T[0, :] = 1.0
        if x.shape[0] > 0:
            T[1, :] = t
            for n in range(2, 10):
                T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
        
        # Linear combination
        result = T @ params
        
        # Ensure finite output
        result = np.where(np.isfinite(result), result, 0.0)
        
        return result
    except Exception:
        # Fallback to zero if anything goes wrong
        return np.zeros_like(x)
