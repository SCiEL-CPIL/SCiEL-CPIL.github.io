import numpy as np

def _legendre_basis(t):
    """
    Compute shifted Legendre polynomials P0..P3 on t in [0,1].
    
    P0(t) = 1
    P1(t) = 2t - 1
    P2(t) = 6t^2 - 6t + 1
    P3(t) = 20t^3 - 30t^2 + 12t - 1
    """
    p0 = np.ones_like(t)
    p1 = 2.0 * t - 1.0
    p2 = 6.0 * t * t - 6.0 * t + 1.0
    p3 = 20.0 * t * t * t - 30.0 * t * t + 12.0 * t - 1.0
    return p0, p1, p2, p3

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Shifted Legendre Polynomial Basis with Cross-Interactions.
    
    This model class assumes the target function can be well-approximated by 
    a polynomial in the normalized inputs. We use shifted Legendre polynomials
    of degree up to 3 for each input, which are orthogonal on [0,1]. This provides
    a well-conditioned basis that captures higher-order curvature.
    
    Inputs are normalized from [0.5, 2] to [0, 1] using t = (val - 0.5) / 1.5.
    
    Basis (10 terms):
    0: P0(x) * P0(y) * P0(z)  (constant 1)
    1: P1(x) * P0(y) * P0(z)  (P1(x))
    2: P0(x) * P1(y) * P0(z)  (P1(y))
    3: P0(x) * P0(y) * P1(z)  (P1(z))
    4: P2(x) * P0(y) * P0(z)  (P2(x))
    5: P0(x) * P2(y) * P0(z)  (P2(y))
    6: P0(x) * P0(y) * P2(z)  (P2(z))
    7: P1(x) * P1(y) * P0(z)  (P1(x)*P1(y))
    8: P1(x) * P0(y) * P1(z)  (P1(x)*P1(z))
    9: P0(x) * P1(y) * P1(z)  (P1(y)*P1(z))
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1]
    tx = (x - 0.5) / 1.5
    ty = (y - 0.5) / 1.5
    tz = (z - 0.5) / 1.5
    
    # Compute Legendre basis for each normalized input
    x0, x1, x2, x3 = _legendre_basis(tx)
    y0, y1, y2, y3 = _legendre_basis(ty)
    z0, z1, z2, z3 = _legendre_basis(tz)
    
    # Linear combination of selected basis terms
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
