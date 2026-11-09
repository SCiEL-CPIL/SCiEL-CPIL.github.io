"""Symbolic regression with a normalized Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_chebyt


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a 10-term Chebyshev series to the data.
    
    The input x is in [0.1, 6]. We map this to [-1, 1] for Chebyshev polynomials.
    The basis functions are normalized by their RMS over the domain to balance
    gradient magnitudes across all 10 parameters, helping BFGS converge better.
    """
    # Map x from [0.1, 6] to [-1, 1]
    # t = (2*x - (x_min + x_max)) / (x_max - x_min)
    # t = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    n_terms = 10
    basis = np.zeros((len(x), n_terms))
    
    if len(x) == 0:
        return np.array([])
    
    basis[:, 0] = 1.0
    if n_terms > 1:
        basis[:, 1] = t
    
    for k in range(2, n_terms):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Normalize each basis function by its RMS over the domain
    # This balances gradient magnitudes and makes the optimization landscape more isotropic
    # We use a fixed normalization based on the domain [0.1, 6]
    # For efficiency, we precompute the RMS values
    
    # For Chebyshev polynomials on [-1, 1], the L2 norm is:
    # ||T_0||^2 = 2, ||T_k||^2 = 1 for k >= 1 (with respect to dt)
    # But our domain is [0.1, 6] mapped to [-1, 1], so we need to account for the Jacobian
    # Actually, let's just compute the RMS empirically using a fine grid
    
    # Precompute normalization constants
    # Using a fine grid for accurate RMS estimation
    x_grid = np.linspace(0.1, 6.0, 1000)
    t_grid = (2.0 * x_grid - 6.1) / 5.9
    
    basis_grid = np.zeros((len(x_grid), n_terms))
    basis_grid[:, 0] = 1.0
    basis_grid[:, 1] = t_grid
    
    for k in range(2, n_terms):
        basis_grid[:, k] = 2.0 * t_grid * basis_grid[:, k-1] - basis_grid[:, k-2]
    
    # Compute RMS for each basis function
    rms = np.sqrt(np.mean(basis_grid ** 2, axis=0))
    
    # Avoid division by zero
    rms = np.where(rms < 1e-12, 1.0, rms)
    
    # Normalize the basis
    basis_normalized = basis / rms
    
    # The model is: y = sum(params[i] * basis_normalized[i])
    return basis_normalized @ params
# EVOLVE-BLOCK-END
