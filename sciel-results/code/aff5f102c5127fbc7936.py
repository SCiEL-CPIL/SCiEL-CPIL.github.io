import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev Polynomial basis.
    Chebyshev polynomials provide a numerically stable basis for polynomial approximation
    over the interval [-1, 1], which is well-conditioned for BFGS optimization compared
    to raw monomials.
    
    Inputs x and v are scaled to [-1, 1] based on the training range.
    
    Terms:
    p0-p3: Chebyshev polynomials T0-T3 of scaled x
    p4-p7: Chebyshev polynomials T0-T3 of scaled v
    p8: Product of T1(x) and T1(v) (linear coupling)
    p9: Product of T2(x) and T2(v) (quadratic coupling)
    """
    # Training range centers and half-ranges
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_c = (x_min + x_max) / 2.0
    x_s = (x_max - x_min) / 2.0
    v_c = (v_min + v_max) / 2.0
    v_s = (v_max - v_min) / 2.0
    
    # Scale inputs to [-1, 1]
    xn = (x - x_c) / x_s
    vn = (v - v_c) / v_s
    
    # Chebyshev polynomials of the first kind
    T0_x = 1.0
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn
    
    T0_v = 1.0
    T1_v = vn
    T2_v = 2.0 * vn**2 - 1.0
    T3_v = 4.0 * vn**3 - 3.0 * vn
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Construct output
    y = (p0 * T0_x + p1 * T1_x + p2 * T2_x + p3 * T3_x +
         p4 * T0_v + p5 * T1_v + p6 * T2_v + p7 * T3_v +
         p8 * T1_x * T1_v +
         p9 * T2_x * T2_v)
         
    return y
