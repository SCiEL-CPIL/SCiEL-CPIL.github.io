import numpy as np
from numpy.polynomial.chebyshev import chebvander

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-2, 2] to [-1, 1]
    # Chebyshev polynomials are defined on [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev basis values of degree 0 to 9
    # chebvander returns a matrix of shape (len(x), 10)
    # Columns correspond to T_0, T_1, ..., T_9
    basis = chebvander(t, 9)
    
    # Linear combination
    return basis @ params
