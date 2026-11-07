# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable product basis: (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) * (c0 + c1*x + c2*y + c3*z)
    
    This expands to a full cubic polynomial in x, y, z with 64 terms, but parameterized
    with only 12 coefficients (a0..a3, b0..b3, c0..c3). However, we only have 10 params.
    
    Alternative: use a structured form that captures interactions with 10 params.
    
    Let's use: (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) + p8*x*y*z + p9*x*y
    This gives us a quadratic form plus specific cubic interaction terms.
    
    Actually, let me think about this more carefully. The product of two linear forms gives
    1 + 3 + 3 + 1 = 9 terms (constant, x, y, z, xy, xz, yz, x^2, y^2, z^2... wait no).
    
    (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) expands to:
    a0*b0 + a0*b1*x + a0*b2*y + a0*b3*z +
    a1*b0*x + a1*b1*x^2 + a1*b2*x*y + a1*b3*x*z +
    a2*b0*y + a2*b1*x*y + a2*b2*y^2 + a2*b3*y*z +
    a3*b0*z + a3*b1*x*z + a3*b2*y*z + a3*b3*z^2
    
    That's 16 terms. With 8 params (a0-a3, b0-b3), we get a quadratic surface.
    
    With 10 params, let's add two more interaction terms:
    pred = (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) + p8*x*y*z + p9*(x+y+z)
    
    Wait, let me just use a clean 10-param model. Let me try:
    
    pred = p0 + p1*x + p2*y + p3*z + p4*x*y + p5*x*z + p6*y*z + p7*x*y*z + p8*x**2 + p9*y**2
    
    This is a full quadratic plus the xyz term. 10 params total.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    pred = (
        p0 +
        p1 * x +
        p2 * y +
        p3 * z +
        p4 * x * y +
        p5 * x * z +
        p6 * y * z +
        p7 * x * y * z +
        p8 * x * x +
        p9 * y * y
    )
    
    return pred
# EVOLVE-BLOCK-END
