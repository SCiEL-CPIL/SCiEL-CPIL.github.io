"""E. coli growth rate model: Cubic Polynomial Response Surface.

Structural change from hybrid biological priors to a pure polynomial basis.
This aligns with findings that scaled polynomials provide a smoother, more
convex-like loss landscape for BFGS, allowing more reliable global convergence
from the 'all ones' start compared to RBFs or constrained biological models.

Strategy:
- Use normalized inputs for numerical stability.
- Include linear, quadratic, and cubic terms for Temperature and pH, which
  are the primary drivers of non-linear biological response (optimal ranges).
- Include linear and quadratic terms for Substrate to capture saturation.
- Include linear term for Population Density.
- No hard-coded biological constants (like Ks or T_opt) to avoid local minima
  traps associated with fixed centers/widths.

Parameter allocation (10 params):
  p[0]: intercept
  p[1]: linear b coefficient
  p[2]: linear s coefficient
  p[3]: quadratic s coefficient
  p[4]: linear temp coefficient
  p[5]: quadratic temp coefficient
  p[6]: cubic temp coefficient
  p[7]: linear pH coefficient
  p[8]: quadratic pH coefficient
  p[9]: cubic pH coefficient
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Model: Cubic polynomial response surface.
    
    Inputs:
        b: population density, range [1, 4.998]
        s: substrate concentration, range [1, 4.999]
        temp: temperature, range [1.002, 39.98]
        pH: pH, range [2, 12]
    
    Parameters:
        params: 10 fitted parameters
    """
    p = params
    
    # Normalize inputs for numerical stability
    # Centering and scaling helps BFGS converge faster and avoids large numbers
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    T_n = (temp - 20.5) / 19.5
    pH_n = (pH - 7.0) / 5.0
    
    # Pre-compute powers
    T2 = T_n * T_n
    T3 = T2 * T_n
    pH2 = pH_n * pH_n
    pH3 = pH2 * pH_n
    s2 = s_n * s_n
    
    # Combine: Polynomial basis
    mu = (p[0]
          + p[1] * b_n
          + p[2] * s_n
          + p[3] * s2
          + p[4] * T_n
          + p[5] * T2
          + p[6] * T3
          + p[7] * pH_n
          + p[8] * pH2
          + p[9] * pH3)
    
    return mu
# EVOLVE-BLOCK-END
