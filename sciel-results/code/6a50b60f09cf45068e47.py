# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Tensor Product of Legendre Polynomials up to total degree 3.
    This uses an orthogonal basis on the domain [-1, 1], which provides
    a well-conditioned linear system for BFGS to solve, avoiding the
    collinearity issues of monomial bases. The 10 parameters correspond
    to the coefficients of the 10 basis functions spanning the space of
    polynomials with total degree <= 3.
    
    Basis functions:
    1. 1
    2. x
    3. y
    4. P2(x) = 0.5 * (3x^2 - 1)
    5. xy
    6. P2(y) = 0.5 * (3y^2 - 1)
    7. P3(x) = 0.5 * (5x^3 - 3x)
    8. x * P2(y)
    9. y * P2(x)
    10. P3(y) = 0.5 * (5y^3 - 3y)
    """
    p = params
    
    # Compute powers of x and y
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    
    # Legendre Polynomials (normalized such that P_n(1)=1, but scaled for stability/orthogonality in LSQ)
    # P0 = 1
    # P1 = x
    # P2 = 0.5 * (3x^2 - 1)
    # P3 = 0.5 * (5x^3 - 3x)
    
    p0 = 1.0
    p1_x = x
    p1_y = y
    
    p2_x = 0.5 * (3.0 * x2 - 1.0)
    p2_y = 0.5 * (3.0 * y2 - 1.0)
    
    p3_x = 0.5 * (5.0 * x3 - 3.0 * x)
    p3_y = 0.5 * (5.0 * y3 - 3.0 * y)
    
    # Construct the linear combination
    # Indices:
    # 0: P0(x)*P0(y) = 1
    # 1: P1(x)*P0(y) = x
    # 2: P0(x)*P1(y) = y
    # 3: P2(x)*P0(y) = P2(x)
    # 4: P1(x)*P1(y) = xy
    # 5: P0(x)*P2(y) = P2(y)
    # 6: P3(x)*P0(y) = P3(x)
    # 7: P1(x)*P2(y) = x * P2(y)
    # 8: P2(x)*P1(y) = P2(x) * y
    # 9: P0(x)*P3(y) = P3(y)
    
    out = np.zeros_like(x, dtype=np.float64)
    
    out += p[0] * p0
    out += p[1] * p1_x
    out += p[2] * p1_y
    out += p[3] * p2_x
    out += p[4] * (x * y)
    out += p[5] * p2_y
    out += p[6] * p3_x
    out += p[7] * (x * p2_y)
    out += p[8] * (p2_x * y)
    out += p[9] * p3_y
    
    return out
# EVOLVE-BLOCK-END
