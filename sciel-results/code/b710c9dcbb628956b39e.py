import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Degree-9 Chebyshev polynomial basis on the scaled domain [-1, 1].
    
    The input x is in [-3, 3]. We scale to t = x/3 in [-1, 1].
    The model is: y(x) = sum_{k=0}^{9} c_k * T_k(t)
    
    Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight 1/sqrt(1-t^2),
    making this a well-conditioned basis for least-squares fitting. Since the model is linear
    in the parameters c_k, BFGS should converge quickly to the global minimum.
    
    Parameters (10 total):
        params[0:10] = c_0, c_1, ..., c_9 (Chebyshev coefficients)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # Using the recurrence relation: T_0 = 1, T_1 = t, T_{n+1} = 2*t*T_n - T_{n-1}
    T = np.zeros((10, len(t)), dtype=np.float64)
    T[0, :] = 1.0
    if len(t) > 0:
        T[1, :] = t
        for k in range(2, 10):
            T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Linear combination: y = sum_k c_k * T_k
    c = params[:10]
    result = T @ c
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
