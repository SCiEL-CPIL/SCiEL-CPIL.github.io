"""Symbolic regression with 1 input - High-degree Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Pure Chebyshev polynomial model of degree 9.
    
    This model is linear in the parameters, ensuring that BFGS converges to the
    global minimum of the least squares problem. Chebyshev polynomials provide
    superior conditioning compared to monomials, enabling reliable convergence
    for any smooth target function on [-3, 3].
    
    10 parameters:
      params[0:10]: Coefficients for Chebyshev polynomials T0..T9 (scaled x)
    
    The Chebyshev basis efficiently captures global smooth trends and is
    well-suited for approximating a wide range of functions.
    """
    # Scale x to [-1, 1] for standard Chebyshev polynomials
    # Original domain is [-3, 3]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T0 to T9 using recurrence relation
    # T0(t) = 1
    # T1(t) = t
    # Tn(t) = 2*t*T(n-1)(t) - T(n-2)(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Build basis matrix
    basis = [T0, T1]
    
    for n in range(2, 10):
        Tn = 2 * t * basis[n-1] - basis[n-2]
        basis.append(Tn)
    
    # Stack basis functions into a matrix of shape (N, 10)
    basis_matrix = np.column_stack(basis)
    
    # Linear combination
    y = basis_matrix @ params
    
    return y
# EVOLVE-BLOCK-END
