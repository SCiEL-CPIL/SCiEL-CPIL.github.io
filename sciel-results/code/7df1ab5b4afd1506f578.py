"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Chebyshev polynomials.
    
    Model: y(x) = sum_{k=0}^{9} c_k * T_k(x/2)
    
    This is a linear model in the parameters, which ensures the optimization
    landscape is convex and BFGS will reliably find the global minimum for
    this basis. Chebyshev polynomials provide a well-conditioned basis for
    approximating smooth functions on the interval [-2, 2].
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Scale input to [-1, 1] to match the domain of Chebyshev polynomials
    t = x / 2.0
    
    # Initialize array to hold Chebyshev polynomial values
    # Shape: (10, N)
    T = np.empty((10, x.size), dtype=np.float64)
    
    # T_0(t) = 1
    T[0] = 1.0
    
    if x.size > 0:
        # T_1(t) = t
        T[1] = t
        
        # Recurrence relation: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        for k in range(2, 10):
            T[k] = 2.0 * t * T[k-1] - T[k-2]
            
    # Compute the linear combination
    # p is shape (10,), T is shape (10, N)
    # np.dot(p, T) results in shape (N,)
    return np.dot(p, T)
# EVOLVE-BLOCK-END
