import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of Chebyshev polynomials.
    
    The model is:
    y = sum_{k=0}^{9} params[k] * T_k(x/3)
    
    where T_k is the Chebyshev polynomial of the first kind of degree k.
    The input x is in [-3, 3], so x/3 is in [-1, 1], the standard domain for T_k.
    
    This model is linear in the parameters, which makes the optimization problem
    convex (quadratic loss). BFGS should converge very rapidly to the optimal
    coefficients. The Chebyshev basis is well-conditioned compared to the power basis,
    ensuring numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # Linear combination
    # params is shape (10,)
    # T is shape (10, N)
    # We want sum(params[k] * T[k, :]) for each point
    result = np.dot(params, T)
    
    return result
