# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Chebyshev polynomial basis (degree 9).
    The basis is linear in parameters, so BFGS converges to the global optimum
    of this flexible function class.
    """
    # Map x from [0.1, 6] to [-1, 1] for stable Chebyshev evaluation
    # t = 2*(x - 0.1)/(6 - 0.1) - 1 = 2*(x - 0.1)/5.9 - 1
    t = (2.0 * (x - 0.1) / 5.9) - 1.0

    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    T = np.empty((10, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n - 1, :] - T[n - 2, :]

    # Linear combination: y = sum_{k=0}^{9} params[k] * T_k(t)
    return T @ params
# EVOLVE-BLOCK-END
