import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function using a Legendre polynomial series expansion.
    Legendre polynomials are orthogonal with respect to the uniform distribution
    on [-1, 1], which matches the input distribution of the data (scaled to [-1, 1]).
    This provides a better conditioned least-squares problem for BFGS optimization
    compared to Chebyshev polynomials, which are orthogonal with respect to a
    different weight function.
    
    Input x is scaled to [-1, 1].
    10 parameters correspond to Legendre coefficients P_0 through P_9.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    n_basis = 10
    n_points = x.size
    
    # Build Legendre basis matrix using recurrence
    # L[:, k] = P_k(u)
    # P_0(u) = 1
    # P_1(u) = u
    # (k+1)P_{k+1}(u) = (2k+1)u P_k(u) - k P_{k-1}(u)
    
    L = np.zeros((n_points, n_basis), dtype=np.float64)
    L[:, 0] = 1.0  # P_0(u) = 1
    
    if n_basis > 1:
        L[:, 1] = u  # P_1(u) = u
        
    for k in range(1, n_basis - 1):
        # Calculate P_{k+1} from P_k and P_{k-1}
        # P_{k+1} = ((2k+1)*u*P_k - k*P_{k-1}) / (k+1)
        L[:, k + 1] = ((2.0 * k + 1.0) * u * L[:, k] - k * L[:, k - 1]) / (k + 1.0)
        
    # Linear combination: y = sum_k params[k] * P_k(u)
    return L @ params
