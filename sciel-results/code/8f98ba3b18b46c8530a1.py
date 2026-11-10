"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining Chebyshev polynomials and Gaussian RBFs.
    
    The first 5 parameters weight Chebyshev polynomials T_0..T_4 (mapped to [-1,1]),
    capturing global smooth trends. The next 5 parameters weight Gaussian RBFs
    centered at evenly spaced points, capturing local variations.
    
    This combination provides both global approximation power (via orthogonal polynomials)
    and local flexibility (via RBFs), which should outperform either basis alone
    for a wider variety of target functions.
    """
    # --- Chebyshev basis (degrees 0-4) ---
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    N = len(x)
    features = np.zeros((N, 10))
    
    # T_0(t) = 1
    features[:, 0] = 1.0
    
    # T_1(t) = t
    features[:, 1] = t
    
    # T_2(t) = 2*t^2 - 1
    features[:, 2] = 2.0 * t * t - 1.0
    
    # T_3(t) = 4*t^3 - 3*t
    features[:, 3] = 2.0 * t * features[:, 2] - features[:, 1]
    
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    features[:, 4] = 2.0 * t * features[:, 3] - features[:, 2]
    
    # --- RBF basis (5 centers) ---
    # Centers evenly spaced across [-3, 3]: -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Width parameter: moderate value for smooth basis functions
    # Using sigma = 1.2 for broader coverage than the original RBF-only model
    sigma = 1.2
    
    # Compute RBF basis: exp(-((x - c)^2) / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Assign RBF values to columns 5-9
    features[:, 5:10] = rbf_values
    
    # Linear combination of features with parameters
    return features @ params
# EVOLVE-BLOCK-END
