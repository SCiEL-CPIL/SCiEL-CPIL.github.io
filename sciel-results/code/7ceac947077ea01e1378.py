import numpy as np
import time

def _shifted_legendre(t, deg):
    """
    Compute shifted Legendre polynomials P_0(t) through P_deg(t) for t in [0, 1].
    Uses the three-term recurrence for numerical stability.
    
    Parameters
    ----------
    t : ndarray
        Values in [0, 1].
    deg : int
        Maximum degree of the polynomial.
    
    Returns
    -------
    ndarray
        Array of shape (deg + 1, N) where N is the size of t.
    """
    p = np.empty((deg + 1, len(t)))
    p[0] = 1.0
    if deg >= 1:
        p[1] = 2.0 * t - 1.0
        for k in range(2, deg + 1):
            # Shifted Legendre recurrence
            p[k] = (2.0 * k + 1.0) * (2.0 * t - 1.0) * p[k-1] - (k - 1.0) * p[k-2]
    return p

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Orthogonal Polynomial Basis using Shifted Legendre Polynomials.
    
    This model uses shifted Legendre polynomials (orthogonal on [0,1]) applied
    to normalized inputs, combined with cross-interaction terms. This provides:
    1. Better numerical conditioning than raw monomials.
    2. Efficient capture of higher-order curvature (up to cubic).
    3. Natural handling of the bounded input domain [0.5, 2] via normalization.
    
    Basis (10 terms):
    0: 1 (constant)
    1: P1(x_norm)
    2: P1(y_norm)
    3: P1(z_norm)
    4: P2(x_norm)
    5: P2(y_norm)
    6: P2(z_norm)
    7: P3(x_norm)
    8: P1(x_norm) * P1(y_norm)
    9: P1(x_norm) * P1(z_norm)
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1]
    x_norm = (x - 0.5) / 1.5
    y_norm = (y - 0.5) / 1.5
    z_norm = (z - 0.5) / 1.5
    
    # Compute shifted Legendre polynomials up to degree 3 for each input
    px = _shifted_legendre(x_norm, 3)  # shape (4, N)
    py = _shifted_legendre(y_norm, 3)  # shape (4, N)
    pz = _shifted_legendre(z_norm, 3)  # shape (4, N)
    
    # Extract individual polynomial terms
    P1_x = px[1]
    P1_y = py[1]
    P1_z = pz[1]
    P2_x = px[2]
    P2_y = py[2]
    P2_z = pz[2]
    P3_x = px[3]
    
    # Cross-interaction terms
    cross_xy = P1_x * P1_y
    cross_xz = P1_x * P1_z
    
    # Linear combination
    return (p[0] + 
            p[1] * P1_x + 
            p[2] * P1_y + 
            p[3] * P1_z + 
            p[4] * P2_x + 
            p[5] * P2_y + 
            p[6] * P2_z + 
            p[7] * P3_x + 
            p[8] * cross_xy + 
            p[9] * cross_xz)
