"""Rewritten program: Separable tensor-product Chebyshev basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a separable tensor-product basis of Chebyshev
    polynomials of the first kind, up to degree 2 in each variable.
    
    Basis:
    - 9 terms: T_i(x) * T_j(y) for i, j in {0, 1, 2}
    - Total 9 parameters used. The 10th parameter is unused (or can be ignored).
    
    This basis is well-conditioned for x, y in [-1, 1] and allows BFGS to
    reliably converge to the global minimum of the quadratic loss function.
    """
    # x: (N,), y: (N,)
    # params: (10,)
    
    # Compute Chebyshev polynomials of the first kind for x and y
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    
    # Construct the 9 basis functions as outer products
    # We map params[0..8] to the 9 basis terms
    # Order: (T0_x*T0_y, T0_x*T1_y, T0_x*T2_y, T1_x*T0_y, T1_x*T1_y, T1_x*T2_y, T2_x*T0_y, T2_x*T1_y, T2_x*T2_y)
    
    p = params[:9]
    
    # Compute weighted sum
    # f = p0*T0x*T0y + p1*T0x*T1y + p2*T0x*T2y + p3*T1x*T0y + p4*T1x*T1y + p5*T1x*T2y + p6*T2x*T0y + p7*T2x*T1y + p8*T2x*T2y
    
    f = (p[0] * T0_x * T0_y + 
         p[1] * T0_x * T1_y + 
         p[2] * T0_x * T2_y + 
         p[3] * T1_x * T0_y + 
         p[4] * T1_x * T1_y + 
         p[5] * T1_x * T2_y + 
         p[6] * T2_x * T0_y + 
         p[7] * T2_x * T1_y + 
         p[8] * T2_x * T2_y)
         
    return f
# EVOLVE-BLOCK-END
