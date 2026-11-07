"""Symbolic regression for E. coli growth rate - Linear Chebyshev basis with interactions."""
import numpy as np


def _chebyshev_basis(x, degree=3):
    """
    Compute Chebyshev polynomial basis functions of given degree.
    x should be in [-1, 1].
    Returns array of shape (..., degree+1) with T0, T1, ..., T_degree.
    """
    # T0 = 1, T1 = x, T2 = 2x^2 - 1, T3 = 4x^3 - 3x
    t0 = np.ones_like(x)
    t1 = x
    t2 = 2.0 * x ** 2 - 1.0
    t3 = 4.0 * x ** 3 - 3.0 * x
    return np.stack([t0, t1, t2, t3], axis=-1)


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model using Chebyshev polynomial basis functions.
    Uses degree-3 Chebyshev polynomials for each input variable plus
    selected interaction terms to capture non-linear relationships.
    
    Parameters (10 total):
        p[0]: bias (constant term)
        p[1]: coefficient for b (linear)
        p[2]: coefficient for s (linear)
        p[3]: coefficient for temp (linear)
        p[4]: coefficient for pH (linear)
        p[5]: coefficient for b*s interaction
        p[6]: coefficient for s*temp interaction
        p[7]: coefficient for temp*pH interaction
        p[8]: coefficient for s^2 (quadratic substrate)
        p[9]: coefficient for temp^2 (quadratic temperature)
    """
    p = params
    
    # Normalize inputs to [-1, 1] range for Chebyshev polynomials
    # b: [1, 4.998] -> [-1, 1]
    b_norm = 2.0 * (b - 1.0) / (4.998 - 1.0) - 1.0
    b_norm = np.clip(b_norm, -1.0, 1.0)
    
    # s: [1, 39.98] -> [-1, 1]
    s_norm = 2.0 * (s - 1.0) / (39.98 - 1.0) - 1.0
    s_norm = np.clip(s_norm, -1.0, 1.0)
    
    # temp: [1, 12] -> [-1, 1]
    temp_norm = 2.0 * (temp - 1.0) / (12.0 - 1.0) - 1.0
    temp_norm = np.clip(temp_norm, -1.0, 1.0)
    
    # pH: [1, 4.999] -> [-1, 1]
    ph_norm = 2.0 * (pH - 1.0) / (4.999 - 1.0) - 1.0
    ph_norm = np.clip(ph_norm, -1.0, 1.0)
    
    # Compute Chebyshev basis for each input
    b_basis = _chebyshev_basis(b_norm)   # (..., 4)
    s_basis = _chebyshev_basis(s_norm)   # (..., 4)
    temp_basis = _chebyshev_basis(temp_norm)  # (..., 4)
    ph_basis = _chebyshev_basis(ph_norm)      # (..., 4)
    
    # Extract linear terms (T1) for main effects
    b_lin = b_basis[..., 1]
    s_lin = s_basis[..., 1]
    temp_lin = temp_basis[..., 1]
    ph_lin = ph_basis[..., 1]
    
    # Extract quadratic terms (T2) for selected interactions
    b_quad = b_basis[..., 2]
    s_quad = s_basis[..., 2]
    temp_quad = temp_basis[..., 2]
    ph_quad = ph_basis[..., 2]
    
    # Build the linear combination
    mu = (
        p[0] * np.ones_like(b) +  # bias
        p[1] * b_lin +            # linear b
        p[2] * s_lin +            # linear s
        p[3] * temp_lin +         # linear temp
        p[4] * ph_lin +           # linear pH
        p[5] * b_lin * s_lin +    # b*s interaction
        p[6] * s_lin * temp_lin + # s*temp interaction
        p[7] * temp_lin * ph_lin + # temp*pH interaction
        p[8] * s_quad +           # quadratic s
        p[9] * temp_quad          # quadratic temp
    )
    
    # Clip to avoid extreme values
    mu = np.clip(mu, -10.0, 10.0)
    
    return mu
