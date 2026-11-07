import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a degree 9 Chebyshev polynomial series with scaled input.
    
    The model is:
        f(x) = sum_{i=0}^{9} params[i] * T_i(x/3.0)
    
    Using Chebyshev polynomials improves the conditioning of the design
    matrix compared to monomial powers, which helps BFGS converge more
    reliably and to a better solution.
    """
    z = x / 3.0
    
    # Ensure z is a 1D array for consistent indexing, though broadcasting works too
    # We need to create a basis matrix of shape (N, 10)
    N = len(z)
    basis = np.empty((N, 10))
    
    basis[:, 0] = 1.0
    basis[:, 1] = z
    for k in range(2, 10):
        basis[:, k] = 2.0 * z * basis[:, k-1] - basis[:, k-2]
        
    y = basis @ params
    return y
