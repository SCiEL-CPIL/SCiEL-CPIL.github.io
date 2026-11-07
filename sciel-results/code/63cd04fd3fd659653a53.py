"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Mixture Model with 3 Gaussians + linear term.
    
    f(x) = a0 + a1*x + sum_{i=1}^{3} A_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2))
    
    Parameters (10 total):
      p[0]  = a0  (intercept)
      p[1]  = a1  (linear slope)
      p[2]  = mu_1 (center of Gaussian 1)
      p[3]  = sigma_1 (width of Gaussian 1, via softplus for positivity)
      p[4]  = A_1  (amplitude of Gaussian 1)
      p[5]  = mu_2 (center of Gaussian 2)
      p[6]  = sigma_2 (width of Gaussian 2, via softplus for positivity)
      p[7]  = A_2  (amplitude of Gaussian 2)
      p[8]  = mu_3 (center of Gaussian 3)
      p[9]  = sigma_3 (width of Gaussian 3, via softplus for positivity)
    
    Note: We need 3 amplitudes + 3 centers + 3 widths + 2 linear = 11 params,
    but we only have 10. So we'll use 2 Gaussians + linear + quadratic:
      p[0] = a0, p[1] = a1, p[2] = a2 (quadratic)
      p[3] = mu_1, p[4] = sigma_1, p[5] = A_1
      p[6] = mu_2, p[7] = sigma_2, p[8] = A_2
      p[9] = A_3 for a third Gaussian with fixed center 0 and fixed width
    
    Actually, let's do: 3 Gaussians with centers, widths, amplitudes = 9 params
    + 1 param for a global linear slope. That's 10.
    
    f(x) = a*x + sum_{i=1}^{3} A_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2))
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    p = p[:10]
    
    # Extract parameters
    a = p[0]  # linear slope
    
    mu1 = p[1]
    sig1 = np.exp(p[2]) + 1e-6  # positive width via exp
    A1 = p[3]
    
    mu2 = p[4]
    sig2 = np.exp(p[5]) + 1e-6  # positive width via exp
    A2 = p[6]
    
    mu3 = p[7]
    sig3 = np.exp(p[8]) + 1e-6  # positive width via exp
    A3 = p[9]
    
    # Compute Gaussian terms
    dx1 = x - mu1
    dx2 = x - mu2
    dx3 = x - mu3
    
    g1 = A1 * np.exp(-0.5 * (dx1 / sig1) ** 2)
    g2 = A2 * np.exp(-0.5 * (dx2 / sig2) ** 2)
    g3 = A3 * np.exp(-0.5 * (dx3 / sig3) ** 2)
    
    y = a * x + g1 + g2 + g3
    
    return y
# EVOLVE-BLOCK-END
