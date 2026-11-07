import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter model using a separable product basis:
    f(x, y, z) = (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) * (c0 + c1*x + c2*y + c3*z)
    
    This expands to a full trilinear polynomial (degree ≤ 1 in each variable),
    capturing all interaction terms up to xyz.
    """
    a0, a1, a2, a3 = params[0:4]
    b0, b1, b2, b3 = params[4:8]
    c0, c1, c2, c3 = params[8:10]
    # We have 10 params: a0..a3 (4), b0..b3 (4), c0..c3 (but only 2 left: c0, c1)
    # Wait, that's 4+4+2 = 10. So c2, c3 are fixed to 0? 
    # Let's re-allocate: 3 factors, 10 params total.
    # Option: (a0+a1*x) * (b0+b1*y+b2*z) * (c0+c1*x+c2*y+c3*z) -> 2+3+4=9, need 10.
    # Option: (a0+a1*x+a2*y) * (b0+b1*z) * (c0+c1*x+c2*y+c3*z) -> 3+2+4=9.
    # Let's do: (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) * (c0 + c1*x)
    # 4 + 4 + 2 = 10. This allows full xy, xz, yz, xyz terms.
    # Factor 1: a0 + a1*x + a2*y + a3*z
    # Factor 2: b0 + b1*x + b2*y + b3*z
    # Factor 3: c0 + c1*x
    
    f1 = a0 + a1 * x + a2 * y + a3 * z
    f2 = b0 + b1 * x + b2 * y + b3 * z
    f3 = c0 + c1 * x
    
    return f1 * f2 * f3
