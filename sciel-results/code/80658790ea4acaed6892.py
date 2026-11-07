"""
Symbolic regression using a Chebyshev polynomial basis.
This approach models y as a linear combination of Chebyshev polynomials of the first kind,
scaled to the input domain [-2, 2]. Chebyshev polynomials are orthogonal on [-1, 1],
which results in a well-conditioned design matrix. This ensures that the optimization
problem (linear least squares) is convex and that scipy.optimize.minimize (BFGS) will
converge rapidly and reliably to the global minimum, avoiding the ill-conditioning
and slow convergence issues associated with monomial or mixed bases.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points in [-2, 2]
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Scale x to [-1, 1] for Chebyshev polynomials
    x_scaled = x / 2.0
    
    # Compute Chebyshev polynomials T_0(x) to T_9(x)
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    
    # We can use numpy's polynomial chebyshev vander for efficiency
    # chebvander(x, deg) returns the Vandermonde matrix of Chebyshev polynomials
    # Shape: (N, deg+1)
    # Columns correspond to T_0, T_1, ..., T_deg
    
    deg = 9
    # x_scaled is 1D array of length N
    # chebvander expects 1D or 2D? It works with 1D.
    V = np.polynomial.chebyshev.chebvander(x_scaled, deg)
    
    # V is N x 10
    # params is 10
    # pred = V @ params
    
    pred = V.dot(params)
    
    return pred
# EVOLVE-BLOCK-END
