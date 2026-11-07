"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev(x, n):
    """Compute Chebyshev polynomial of the first kind of degree n for array x in [-1,1]."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    c0 = np.ones_like(x)
    c1 = x.copy()
    for _ in range(2, n + 1):
        c2 = 2.0 * x * c1 - c0
        c0 = c1
        c1 = c2
    return c1


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters Chebyshev polynomial basis model for E. coli growth rate.
    
    Uses a complete set of monomials up to total degree 3 in 4 normalized inputs,
    with Chebyshev polynomials in each input for numerical stability.
    Linear model ensures BFGS converges to global minimum.
    """
    # Normalize inputs to [-1, 1] for Chebyshev stability
    b_n = (b - 3.0) / 2.0       # [1, 5] -> [-1, 1]
    s_n = (s - 20.5) / 19.5     # [1, 40] -> [-1, 1]
    t_n = (temp - 6.5) / 5.5    # [1, 12] -> [-1, 1]
    p_n = (pH - 3.0) / 2.0      # [1, 5] -> [-1, 1]
    
    # Clip to [-1, 1] for numerical safety
    b_n = np.clip(b_n, -1.0, 1.0)
    s_n = np.clip(s_n, -1.0, 1.0)
    t_n = np.clip(t_n, -1.0, 1.0)
    p_n = np.clip(p_n, -1.0, 1.0)
    
    # Precompute Chebyshev polynomials up to degree 3 for each input
    b0, b1, b2, b3 = _chebyshev(b_n, 0), _chebyshev(b_n, 1), _chebyshev(b_n, 2), _chebyshev(b_n, 3)
    s0, s1, s2, s3 = _chebyshev(s_n, 0), _chebyshev(s_n, 1), _chebyshev(s_n, 2), _chebyshev(s_n, 3)
    t0, t1, t2, t3 = _chebyshev(t_n, 0), _chebyshev(t_n, 1), _chebyshev(t_n, 2), _chebyshev(t_n, 3)
    p0, p1, p2, p3 = _chebyshev(p_n, 0), _chebyshev(p_n, 1), _chebyshev(p_n, 2), _chebyshev(p_n, 3)
    
    # Build feature matrix with all monomials up to total degree 3
    # We use 10 parameters, so we select the most important features
    # Degree 0: 1 feature
    # Degree 1: 4 features (b, s, t, p)
    # Degree 2: 6 features (b^2, s^2, t^2, p^2, bs, bt, bp, st, sp, tp) - pick 4 most likely
    # Degree 3: select key interactions
    
    # Feature selection for 10 parameters:
    # [0] intercept
    # [1] b (biomass)
    # [2] s (substrate) 
    # [3] t (temperature)
    # [4] p (pH)
    # [5] s*t (substrate-temperature interaction)
    # [6] s*p (substrate-pH interaction)
    # [7] t*p (temperature-pH interaction)
    # [8] s^2 (substrate saturation)
    # [9] t^2 (temperature optimum curvature)
    
    mu = (params[0] * b0 * s0 * t0 * p0 +
          params[1] * b1 * s0 * t0 * p0 +
          params[2] * s1 * b0 * t0 * p0 +
          params[3] * t1 * b0 * s0 * p0 +
          params[4] * p1 * b0 * s0 * t0 +
          params[5] * s1 * t1 * b0 * p0 +
          params[6] * s1 * p1 * b0 * t0 +
          params[7] * t1 * p1 * b0 * s0 +
          params[8] * s2 * b0 * t0 * p0 +
          params[9] * t2 * b0 * s0 * p0)
    
    return mu
# EVOLVE-BLOCK-END
