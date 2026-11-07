"""
Rewritten program for symbolic regression.
Uses a Chebyshev series of degree 9.
This model is linear in the parameters, ensuring the optimization landscape is convex.
This allows scipy.optimize.minimize (BFGS) to reliably find the global least-squares solution,
avoiding the local minima pitfalls of non-linear basis functions like Tanh.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard variables
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev series of degree 9.
    
    Model:
    y = sum_{k=0}^{9} p_k * T_k(x/2)
    
    where T_k is the k-th Chebyshev polynomial of the first kind.
    The input x is scaled to [-1, 1] by dividing by 2 to ensure
    the Chebyshev polynomials are well-conditioned.
    
    Parameters:
    p0..p9: Coefficients for T_0..T_9
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # If we hit the deadline, we just return the current calculation
        # which is instantaneous anyway.
        pass

    # Scale input to [-1, 1] for better numerical conditioning of Chebyshev polys
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Linear combination: y = sum(p_k * T_k)
    # params is shape (10,)
    # T is shape (10, N)
    # We want shape (N,)
    y = np.sum(params[:, np.newaxis] * T, axis=0)
    
    return y
# EVOLVE-BLOCK-END
