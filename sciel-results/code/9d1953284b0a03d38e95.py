"""
Symbolic regression: Chebyshev-based product-of-sums with interaction terms.
Combines the multiplicative structure of the current program with the
orthogonal Chebyshev basis for better conditioning and the trigonometric
flexibility from inspirations for non-polynomial shapes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10. The model uses a hybrid approach:
    
    1. A product-of-sums term using Chebyshev polynomials of degree 0-2 in x
       and y, which captures multiplicative interactions with good conditioning.
       (p0 + p1*T1(x) + p2*T2(x)) * (p3 + p4*T1(y) + p5*T2(y))
       
    2. Additional interaction and higher-order terms:
       p6 * T1(x)*T1(y)  (cross term)
       p7 * T2(x)*T1(y)  (higher order interaction)
       p8 * T1(x)*T2(y)  (higher order interaction)
       p9 * sin(pi*x)*sin(pi*y)  (trigonometric interaction for oscillatory behavior)
    
    Chebyshev polynomials are orthogonal on [-1,1], which improves the
    conditioning of the design matrix for BFGS optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute Chebyshev polynomials (T0=1, T1=x, T2=2x^2-1)
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    
    # Product of two linear forms in Chebyshev basis
    left = p0 * T0_x + p1 * T1_x + p2 * T2_x
    right = p3 * T0_y + p4 * T1_y + p5 * T2_y
    product_term = left * right
    
    # Additional interaction terms
    interaction = (p6 * T1_x * T1_y + 
                   p7 * T2_x * T1_y + 
                   p8 * T1_x * T2_y +
                   p9 * np.sin(np.pi * x) * np.sin(np.pi * y))
    
    pred = product_term + interaction
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something went wrong
        pred = p0 * T0_x + p1 * T1_x + p2 * T1_y
    
    return pred
# EVOLVE-BLOCK-END
