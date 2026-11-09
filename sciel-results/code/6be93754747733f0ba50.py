# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: RBF kernels + polynomial + trigonometric terms.
    10 parameters:
      params[0..3]: weights for 4 Gaussian RBF centers
      params[4..6]: polynomial coefficients (const, linear, quadratic)
      params[7..9]: trigonometric coefficients (sin(x), cos(x), sin(2x))
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Fixed RBF centers spread across [-3, 3]
    centers = np.array([-2.0, -0.5, 1.0, 2.5])
    bandwidth = 1.2  # sigma for Gaussian
    
    # RBF terms: exp(-((x - c)^2) / (2*sigma^2))
    rbf_terms = np.zeros((n, 4))
    for i, c in enumerate(centers):
        rbf_terms[:, i] = np.exp(-((x - c) ** 2) / (2.0 * bandwidth ** 2))
    
    # Polynomial terms (scaled x for conditioning)
    t = x / 3.0
    poly_const = np.ones(n)
    poly_lin = t
    poly_quad = t * t
    
    # Trigonometric terms
    trig_sin = np.sin(x)
    trig_cos = np.cos(x)
    trig_sin2 = np.sin(2.0 * x)
    
    # Assemble prediction
    y = (
        params[0] * rbf_terms[:, 0] +
        params[1] * rbf_terms[:, 1] +
        params[2] * rbf_terms[:, 2] +
        params[3] * rbf_terms[:, 3] +
        params[4] * poly_const +
        params[5] * poly_lin +
        params[6] * poly_quad +
        params[7] * trig_sin +
        params[8] * trig_cos +
        params[9] * trig_sin2
    )
    
    return y
# EVOLVE-BLOCK-END
