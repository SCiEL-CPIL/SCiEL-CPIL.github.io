"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on [-3, 3].
    Maps x to [-1, 1] and evaluates T_0(x) through T_9(x).
    params[0..9] are the coefficients for each Chebyshev polynomial.
    """
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0

    # Evaluate Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    basis = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * T1 - basis[-1]
        basis.append(Tn)
        T1 = Tn

    # Stack basis: shape (10, n_points)
    basis_matrix = np.vstack(basis)  # (10, N)
    
    # params has length 10, use first 10
    coeffs = params[:10]
    
    # Result: (10,) @ (10, N) -> (N,)
    return coeffs @ basis_matrix
# EVOLVE-BLOCK-END
