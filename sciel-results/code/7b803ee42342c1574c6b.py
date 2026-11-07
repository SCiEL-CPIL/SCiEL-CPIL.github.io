import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combined Hybrid Basis: Quadratic Polynomial + Fundamental Sine Terms.
    
    This program combines the structural idea of the full quadratic polynomial basis
    (from inspirations 1 & 2, which span the space of degree <= 2 polynomials)
    with the trigonometric components from the current program.
    
    The current program (score ~4.45) uses:
      [1, x, y, z, xy, xz, yz, sin(pi*x), sin(pi*y), sin(pi*z)]
    
    The quadratic basis (score ~7.39) uses:
      [1, x, y, z, x^2, xy, xz, y^2, yz, z^2]
    
    The best parent (score ~1.29) used:
      [1, x, y, z, x^2, y^2, z^2, sin(pi*x), sin(pi*y), sin(pi*z)]
    
    Key insight: The parent's score of ~1.29 was achieved with a mix of linear,
    self-quadratic, and sine terms. The current program replaced self-quadratic
    with cross-interaction terms and scored worse (~4.45). This suggests that
    self-quadratic terms are more important than cross-interaction terms for this
    hidden function, OR that the sine terms are crucial.
    
    However, we have only 10 parameters. We cannot include all of:
      1, x, y, z, x^2, y^2, z^2, xy, xz, yz, sin(pi*x), sin(pi*y), sin(pi*z)
    
    We need to make a strategic choice. The parent's success with 
    [1, x, y, z, x^2, y^2, z^2, sin(pi*x), sin(pi*y), sin(pi*z)]
    suggests that self-quadratic + sine is a good combination.
    
    But wait - the current program is the one we are improving. It has score 4.45.
    The parent had score 1.29. So we should go back to a basis closer to the parent,
    but perhaps refine it.
    
    Let's try a basis that includes both self-quadratic AND some interaction terms,
    but drops one sine term or combines them differently.
    
    Actually, let's think about what the hidden function might be.
    If it's a smooth function on [0.5, 2]^3, it could be:
    - A polynomial of degree 2 or 3
    - A trigonometric function
    - A combination
    
    The parent's basis [1, x, y, z, x^2, y^2, z^2, sin(pi*x), sin(pi*y), sin(pi*z)]
    achieved ~1.29. This is a mix of polynomial and trig.
    
    Let's try a basis that includes:
    1, x, y, z, x*y, x*z, y*z, sin(pi*x), sin(pi*y), sin(pi*z)
    This is the current program's basis, which scored 4.45.
    
    Or:
    1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z
    This is the full quadratic, which scored ~7.39.
    
    Or:
    1, x, y, z, x^2, y^2, z^2, sin(pi*x), sin(pi*y), sin(pi*z)
    This is the parent, which scored ~1.29.
    
    Since the parent scored better, let's use a basis close to the parent but with
    a structural refinement. Perhaps we can use a mix that includes one interaction
    term and drops one sine term, or use higher-order sine terms.
    
    Let's try:
    1, x, y, z, x^2, y^2, z^2, sin(pi*x), sin(pi*y), sin(pi*z)
    But with a different optimization strategy or basis normalization.
    
    Actually, the key insight from the inspirations is that using an orthogonal
    basis (like Legendre polynomials) can improve conditioning. Let's combine
    the parent's basis choice with Legendre polynomial normalization for the
    polynomial part.
    
    New Basis:
    1, x, y, z, P2(x'), P2(y'), P2(z'), sin(pi*x), sin(pi*y), sin(pi*z)
    where x' = (2x - 2.5)/1.5 maps [0.5, 2] to [-1, 1]
    and P2(t) = 0.5*(3t^2 - 1)
    
    This maintains the same functional space as the parent for the polynomial part
    (since P2 is a linear transformation of x^2), but with better conditioning.
    """
    p = params
    
    # Map inputs to [-1, 1] for better conditioning
    # x in [0.5, 2] -> u in [-1, 1]
    u = (2.0 * x - 2.5) / 1.5
    v = (2.0 * y - 2.5) / 1.5
    w = (2.0 * z - 2.5) / 1.5
    
    # Legendre P2(t) = 0.5 * (3*t^2 - 1)
    p2_u = 0.5 * (3.0 * u * u - 1.0)
    p2_v = 0.5 * (3.0 * v * v - 1.0)
    p2_w = 0.5 * (3.0 * w * w - 1.0)
    
    # Trigonometric components
    sin_pi_x = np.sin(np.pi * x)
    sin_pi_y = np.sin(np.pi * y)
    sin_pi_z = np.sin(np.pi * z)
    
    return (p[0]
            + p[1] * x
            + p[2] * y
            + p[3] * z
            + p[4] * p2_u
            + p[5] * p2_v
            + p[6] * p2_w
            + p[7] * sin_pi_x
            + p[8] * sin_pi_y
            + p[9] * sin_pi_z)
