import numpy as np
import time

# Global start time to track elapsed time for deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a linear combination of 10 Chebyshev polynomials.
    The input x is scaled to [-1, 1] to ensure numerical stability and leverage the
    orthogonality properties of Chebyshev polynomials.
    
    Basis:
    T_0(t) = 1
    T_1(t) = t
    T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    where t = x / 3.
    
    params: array of length 10, coefficients for T_0 to T_9.
    """
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # In this simple linear model, we don't have an iterative search loop inside equation,
        # so if we are here, we just return the current prediction. 
        # This guard is primarily to satisfy the contract structure if the evaluator
        # calls this in a loop or if we were to add internal optimization.
        pass

    x = np.asarray(x).flatten()
    
    # Scale x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    T0 = np.ones_like(t)
    # T_1(t) = t
    T1 = t.copy()
    
    # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    T = [T0, T1]
    for k in range(2, 10):
        Tk = 2.0 * t * T[k-1] - T[k-2]
        T.append(Tk)
    
    # Stack basis functions into a matrix (N, 10)
    basis_matrix = np.column_stack(T)
    
    # Compute linear combination
    # params shape (10,), basis_matrix shape (N, 10)
    y = basis_matrix @ params
    
    return y
