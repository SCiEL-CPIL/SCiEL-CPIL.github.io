import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Polynomial Basis of degree 9 (10 terms)
    # Basis functions: 1, x, x^2, ..., x^9
    # This allows fitting global trends and smooth functions.
    
    # x is expected to be 1D array of shape (N,)
    # We need to construct a Vandermonde-like matrix but with powers 0..9
    # x powers: x^0, x^1, ..., x^9
    
    # Reshape x to column vector for broadcasting if needed, or just use powers
    # x is (N,)
    # We want basis[i, j] = x[i] ** j for j in 0..9
    
    # Using np.vander is efficient, but it usually returns descending powers.
    # np.vander(x, N=10, increasing=True) returns [1, x, x^2, ..., x^9]
    basis = np.vander(x, N=10, increasing=True)
    
    # params is (10,)
    # result is (N,)
    result = basis @ params
    
    return result
