"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich, well-conditioned basis of 10 terms.
    
    The basis is designed to be well-conditioned and cover a wide range of
    possible target functions:
    - Chebyshev polynomials T_0 through T_4 (5 terms) for polynomial-like behavior
    - sin(x), cos(x) for periodic behavior
    - sin(2x), cos(2x) for higher-frequency periodic behavior
    - exp(-x^2) for a Gaussian bump (localized non-polynomial)
    - tanh(x/3) for saturation behavior
    
    The domain is [-3, 3]. Chebyshev polynomials are evaluated on the mapped
    variable t = x/3 in [-1, 1] for numerical stability.
    
    The 10 parameters are fitted by the evaluator's BFGS optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_4(t)
    T0 = np.ones_like(t_safe)
    T1 = t_safe.copy()
    T2 = 2.0 * t_safe * T1 - T0
    T3 = 2.0 * t_safe * T2 - T1
    T4 = 2.0 * t_safe * T3 - T2
    
    # Trigonometric terms (using x directly, since domain is [-3,3])
    sin1 = np.sin(x)
    cos1 = np.cos(x)
    
    # Localized Gaussian Radial Basis Functions
    # Center at 0
    gauss_c = np.exp(-t_safe**2)
    # Centers at +/- 1.5 (mapped to t = +/- 0.5)
    # t = x/3, so x=1.5 => t=0.5
    gauss_r = np.exp(-(t_safe - 0.5)**2)
    gauss_l = np.exp(-(t_safe + 0.5)**2)
    
    # Saturation
    tanh_term = np.tanh(t_safe)
    
    # Linear combination of 10 terms
    # We drop sin(2x), cos(2x) and T4 to make room for the extra Gaussians?
    # Current count: T0,T1,T2,T3 (4) + sin1,cos1 (2) + gauss_c, gauss_r, gauss_l (3) + tanh (1) = 10.
    # This removes T4. T4 is a higher order polynomial. The Gaussians might capture non-polynomial shapes better.
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * sin1 +
              params[5] * cos1 +
              params[6] * gauss_c +
              params[7] * gauss_r +
              params[8] * gauss_l +
              params[9] * tanh_term)
    
    # Ensure the result is finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
