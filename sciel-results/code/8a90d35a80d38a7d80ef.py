import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    # t = (2 * x - (0.5 + 2)) / (2 - 0.5) = (2*x - 2.5) / 1.5
    tx = (2.0 * x - 2.5) / 1.5
    ty = (2.0 * y - 2.5) / 1.5
    tz = (2.0 * z - 2.5) / 1.5
    
    # Compute Chebyshev polynomials T0, T1, T2, T3 for each variable
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2*t^2 - 1
    # T3(t) = 4*t^3 - 3*t
    
    # We have 10 parameters. We need 10 basis functions.
    # A common strategy is to include:
    # 1. Constant (T0 for all)
    # 2. Linear terms (T1)
    # 3. Interaction terms (products of T1s)
    # 4. Quadratic terms (T2)
    # 5. Higher order interactions
    
    # Let's construct a basis that covers low-order interactions well.
    # Basis:
    # 0: 1 (Constant)
    # 1: tx (x)
    # 2: ty (y)
    # 3: tz (z)
    # 4: tx*ty (xy)
    # 5: ty*tz (yz)
    # 6: tz*tx (zx)
    # 7: tx*ty*tz (xyz)
    # 8: tx^2 + ty^2 + tz^2 (or just one quadratic, but sum is symmetric)
    #    Actually, let's use specific Chebyshev terms to be more flexible.
    #    Let's use:
    #    0: 1
    #    1: tx
    #    2: ty
    #    3: tz
    #    4: tx*ty
    #    5: ty*tz
    #    6: tz*tx
    #    7: tx*ty*tz
    #    8: T2(tx) + T2(ty) + T2(tz)  (Quadratic sum)
    #    9: T3(tx) + T3(ty) + T3(tz)  (Cubic sum)
    
    # This basis allows fitting:
    # - Constants
    # - Linear dependencies
    # - Pairwise interactions
    # - Triple interactions
    # - Non-linear single-variable dependencies (via T2, T3 sums)
    
    b0 = np.ones_like(x)
    b1 = tx
    b2 = ty
    b3 = tz
    b4 = tx * ty
    b5 = ty * tz
    b6 = tz * tx
    b7 = tx * ty * tz
    
    # T2(t) = 2t^2 - 1
    t2x = 2.0 * tx * tx - 1.0
    t2y = 2.0 * ty * ty - 1.0
    t2z = 2.0 * tz * tz - 1.0
    b8 = t2x + t2y + t2z
    
    # T3(t) = 4t^3 - 3t
    t3x = 4.0 * tx * tx * tx - 3.0 * tx
    t3y = 4.0 * ty * ty * ty - 3.0 * ty
    t3z = 4.0 * tz * tz * tz - 3.0 * tz
    b9 = t3x + t3y + t3z
    
    # Construct result as linear combination
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
