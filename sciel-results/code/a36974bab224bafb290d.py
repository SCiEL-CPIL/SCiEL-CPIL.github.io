import numpy as np

def _chebyshev_basis(t):
    """
    Compute Chebyshev polynomials of the first kind T0..T3 on t in [-1, 1].
    
    T0(t) = 1
    T1(t) = t
    T2(t) = 2t^2 - 1
    T3(t) = 4t^3 - 3t
    """
    t0 = np.ones_like(t)
    t1 = t.copy()
    t2 = 2.0 * t * t - 1.0
    t3 = 4.0 * t * t * t - 3.0 * t
    return t0, t1, t2, t3

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev Polynomial Basis (separable, up to degree 3 per variable).
    
    Uses Chebyshev polynomials of the first kind mapped from [0.5, 2] to [-1, 1].
    Chebyshev polynomials minimize the maximum approximation error on the interval,
    providing faster convergence for smooth target functions than Legendre polynomials.
    
    Basis (10 terms):
    0: T0 = 1 (constant)
    1: T1(x)
    2: T1(y)
    3: T1(z)
    4: T2(x)
    5: T2(y)
    6: T2(z)
    7: T3(x)
    8: T3(y)
    9: T3(z)
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [-1, 1]
    tx = (2.0 * (x - 0.5) - 1.5) / 1.5  # maps [0.5,2] to [-1,1]
    ty = (2.0 * (y - 0.5) - 1.5) / 1.5
    tz = (2.0 * (z - 0.5) - 1.5) / 1.5
    
    # Compute Chebyshev polynomial values
    _, t1x, t2x, t3x = _chebyshev_basis(tx)
    _, t1y, t2y, t3y = _chebyshev_basis(ty)
    _, t1z, t2z, t3z = _chebyshev_basis(tz)
    
    # Linear combination
    return (p[0] + 
            p[1] * t1x + 
            p[2] * t1y + 
            p[3] * t1z + 
            p[4] * t2x + 
            p[5] * t2y + 
            p[6] * t2z + 
            p[7] * t3x + 
            p[8] * t3y + 
            p[9] * t3z)
