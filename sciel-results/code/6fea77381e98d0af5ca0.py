import numpy as np

def _shifted_legendre(t, deg):
    """
    Compute shifted Legendre polynomials P_0(t) through P_deg(t) for t in [0, 1].
    Uses the standard three-term recurrence for numerical stability.
    
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
    N = len(t)
    p = np.empty((deg + 1, N))
    p[0] = 1.0
    if deg >= 1:
        p[1] = 2.0 * t - 1.0
        for k in range(2, deg + 1):
            # Recurrence: P_k(t) = ((2k-1)(2t-1)P_{k-1}(t) - (k-1)P_{k-2}(t)) / k
            p[k] = ((2.0 * k - 1.0) * (2.0 * t - 1.0) * p[k-1] - (k - 1.0) * p[k-2]) / k
    return p

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full Quadratic Polynomial Basis using Shifted Legendre Polynomials.
    
    This model assumes the target function can be well-approximated by a polynomial
    of total degree up to 2. It uses shifted Legendre polynomials for better
    numerical conditioning and orthogonality.
    
    Basis (10 terms):
    0: P0(x) P0(y) P0(z)  (Constant)
    1: P1(x) P0(y) P0(z)  (Linear x)
    2: P0(x) P1(y) P0(z)  (Linear y)
    3: P0(x) P0(y) P1(z)  (Linear z)
    4: P2(x) P0(y) P0(z)  (Quadratic x)
    5: P0(x) P2(y) P0(z)  (Quadratic y)
    6: P0(x) P0(y) P2(z)  (Quadratic z)
    7: P1(x) P1(y) P0(z)  (Interaction xy)
    8: P1(x) P0(y) P1(z)  (Interaction xz)
    9: P0(x) P1(y) P1(z)  (Interaction yz)
    
    This basis spans the same function space as a full quadratic monomial basis
    (1, x, y, z, x^2, y^2, z^2, xy, xz, yz) but with superior conditioning.
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1] for Legendre basis
    tx = (x - 0.5) / 1.5
    ty = (y - 0.5) / 1.5
    tz = (z - 0.5) / 1.5
    
    # Compute Legendre basis for each normalized input up to degree 2
    x0, x1, x2 = _shifted_legendre(tx, 2)
    y0, y1, y2 = _shifted_legendre(ty, 2)
    z0, z1, z2 = _shifted_legendre(tz, 2)
    
    # Linear combination of the 10 basis terms
    return (p[0] * x0 * y0 * z0 + 
            p[1] * x1 * y0 * z0 + 
            p[2] * x0 * y1 * z0 + 
            p[3] * x0 * y0 * z1 + 
            p[4] * x2 * y0 * z0 + 
            p[5] * x0 * y2 * z0 + 
            p[6] * x0 * y0 * z2 + 
            p[7] * x1 * y1 * z0 + 
            p[8] * x1 * y0 * z1 + 
            p[9] * x0 * y1 * z1)
