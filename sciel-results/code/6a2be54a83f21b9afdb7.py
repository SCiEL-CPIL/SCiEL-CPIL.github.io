"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf

# Pre-compute RBF basis parameters
# Domain [0.1, 6.0]
X_MIN = 0.1
X_MAX = 6.0
N_PARAMS = 10
SIGMA = 0.8

# Centers uniformly spaced
CENTERS = np.linspace(X_MIN, X_MAX, N_PARAMS)

# Pre-compute mean and std for each basis function phi_i(x) = exp(-(x-c_i)^2 / (2*sigma^2))
# for x ~ Uniform[X_MIN, X_MAX]
# E[phi] = (1/(X_MAX-X_MIN)) * integral
# E[phi^2] = (1/(X_MAX-X_MIN)) * integral of exp(-(x-c)^2 / sigma^2)
# Note: phi^2 = exp(-(x-c)^2 / sigma^2) = exp(-(x-c)^2 / (2*(sigma/sqrt(2))^2))
# So E[phi^2] uses sigma_eff = sigma / sqrt(2)

def _compute_rbf_moments(center, sigma_val, x_min, x_max):
    # Integral of exp(-(x-c)^2 / (2*s^2)) from x_min to x_max
    # = sqrt(pi/2) * s * (erf((x_max-c)/(s*sqrt(2))) - erf((x_min-c)/(s*sqrt(2))))
    s = sigma_val
    arg_max = (x_max - center) / (s * np.sqrt(2))
    arg_min = (x_min - center) / (s * np.sqrt(2))
    integral = np.sqrt(np.pi / 2.0) * s * (erf(arg_max) - erf(arg_min))
    mean = integral / (x_max - x_min)
    
    # For variance, we need E[phi^2]
    # phi^2 = exp(-(x-c)^2 / sigma^2)
    # This is a Gaussian with sigma_eff = sigma / sqrt(2)
    s_eff = sigma_val / np.sqrt(2.0)
    arg_max2 = (x_max - center) / (s_eff * np.sqrt(2))
    arg_min2 = (x_min - center) / (s_eff * np.sqrt(2))
    integral2 = np.sqrt(np.pi / 2.0) * s_eff * (erf(arg_max2) - erf(arg_min2))
    mean_sq = integral2 / (x_max - x_min)
    
    var = mean_sq - mean**2
    if var < 1e-12:
        var = 1e-12
    std = np.sqrt(var)
    return mean, std

MEANS = np.zeros(N_PARAMS)
STDS = np.ones(N_PARAMS)
for i in range(N_PARAMS):
    MEANS[i], STDS[i] = _compute_rbf_moments(CENTERS[i], SIGMA, X_MIN, X_MAX)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a linear combination of Gaussian Radial Basis Functions (RBFs) 
    with fixed centers and widths. The model is linear in the parameters, 
    allowing the external BFGS optimizer to find the global least-squares 
    solution. The basis is standardized using pre-computed moments for the 
    uniform input distribution to ensure well-conditioned optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis matrix
    # x shape (N, 1), CENTERS shape (10,)
    # diff shape (N, 10)
    diff = x[:, np.newaxis] - CENTERS[np.newaxis, :]
    
    # exp(-(diff^2) / (2 * sigma^2))
    basis = np.exp(-(diff ** 2) / (2.0 * SIGMA ** 2))
    
    # Standardize
    basis_std = (basis - MEANS) / STDS
    
    return basis_std @ params
# EVOLVE-BLOCK-END
