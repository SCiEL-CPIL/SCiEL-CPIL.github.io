"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Compute Chebyshev polynomials T_0(z), T_1(z), ..., T_degree(z).
    
    Uses the recurrence: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    
    Returns array of shape (N, degree+1).
    """
    N = z.shape[0]
    basis = np.empty((N, degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = z
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * z * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials.
    
    f(x) = (sum_{i=0}^{5} a_i * T_i(z)) / (sum_{j=0}^{3} b_j * T_j(z))
    where z = x/2 (normalized to [-1,1])
    
    params[0:6] = a_0, a_1, a_2, a_3, a_4, a_5 (numerator coefficients)
    params[6:10] = b_0, b_1, b_2, b_3 (denominator coefficients)
    
    To avoid division by zero, we add a small epsilon to the denominator.
    """
    # Normalize input from [-2, 2] to [-1, 1] for stable Chebyshev evaluation
    z = x / 2.0
    
    # Build Chebyshev basis for numerator (degree 5) and denominator (degree 3)
    num_basis = _chebyshev_basis(z, 5)   # shape (N, 6)
    den_basis = _chebyshev_basis(z, 3)   # shape (N, 4)
    
    # Extract parameters
    a = params[0:6]
    b = params[6:10]
    
    # Compute numerator and denominator
    numerator = num_basis @ a  # shape (N,)
    denominator = den_basis @ b  # shape (N,)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    denominator_safe = denominator + eps * np.sign(denominator)
    # If denominator is zero, use eps with sign of a small positive
    # Actually, let's be more careful:
    mask = np.abs(denominator) < 1e-10
    denominator_safe = denominator.copy()
    denominator_safe[mask] = 1e-8  # small positive value where denominator is near zero
    
    result = numerator / denominator_safe
    
    # Replace non-finite values with large finite values to avoid breaking the optimizer
    result = np.where(np.isfinite(result), result, 1e6)
    
    return result
# EVOLVE-BLOCK-END
