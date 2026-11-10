"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial.chebyshev import chebvander


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial expansion of degree 9.
    
    The input x is mapped from [-2, 2] to [-1, 1] to utilize the standard Chebyshev
    polynomials. The 10 parameters correspond to the coefficients of T_0(t) through T_9(t).
    
    This model is linear in the parameters, ensuring the optimization problem (minimizing
    MSE) is convex. This allows BFGS to reliably find the global minimum, avoiding the
    local minima and instability issues associated with non-convex models like rational
    functions or unstructured non-linear basis mixes. Chebyshev polynomials provide a
    well-conditioned basis for approximation on bounded intervals, offering a significant
    structural improvement over the simple linear model.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute the Chebyshev basis matrix V where V[i, j] = T_j(t_i) for j=0..9
    # chebvander(t, 9) returns a matrix of shape (N, 10)
    V = chebvander(t, 9)
    
    # params is an array of length 10
    # Return the linear combination of basis functions
    return V @ params
# EVOLVE-BLOCK-END
