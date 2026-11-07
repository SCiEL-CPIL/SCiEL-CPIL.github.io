import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using an orthonormal Legendre polynomial basis.
    
    Legendre polynomials are orthogonal with respect to the uniform weight on [-1, 1].
    By normalizing them to have unit norm, we ensure that the parameter space is well-conditioned.
    The Hessian of the MSE loss with respect to the parameters is approximately diagonal (identity),
    which allows BFGS to converge rapidly and reliably, avoiding the conditioning issues
    that likely caused the previous Chebyshev-based attempt to fail.
    """
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Legendre polynomials P_k(t) for k=0..9
    # P_0(t) = 1
    # P_1(t) = t
    # (k+1)P_{k+1}(t) = (2k+1)tP_k(t) - kP_{k-1}(t)
    
    P = [np.ones_like(t), t]
    for k in range(1, 9):
        # Calculate P_{k+1}
        # P_{k+1} = ((2k+1)*t*P_k - k*P_{k-1}) / (k+1)
        P_next = ((2.0 * k + 1.0) * t * P[k] - k * P[k-1]) / (k + 1.0)
        P.append(P_next)
        
    # Normalize each polynomial to have unit norm in the continuous sense
    # Norm of P_k on [-1, 1] is sqrt(2 / (2k + 1))
    # We divide P_k by this norm to get orthonormal basis functions
    
    basis = []
    for k in range(10):
        norm = np.sqrt(2.0 / (2.0 * k + 1.0))
        basis.append(P[k] / norm)
        
    # Stack basis functions: shape (n_points, 10)
    # basis[i] is the i-th basis function evaluated at all points x
    # We want result = sum(params[i] * basis[i])
    # So we stack along the last axis to get (n_points, 10)
    basis_matrix = np.stack(basis, axis=-1)
    
    # Linear combination
    result = basis_matrix @ params
    
    return result
