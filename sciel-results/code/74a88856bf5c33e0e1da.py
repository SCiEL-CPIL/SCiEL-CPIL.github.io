import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration using physically-motivated terms with tanh saturation.
    This structure improves BFGS convergence by providing a better-conditioned
    parameter landscape compared to raw polynomials or Chebyshev expansions.
    
    Inputs x and v are normalized to O(1) range.
    Terms include linear/nonlinear restoring forces, damping, and cross-couplings.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to improve conditioning
    # x range: [-0.5728, 0.5275] -> mid: -0.02265, half_range: 0.55015
    # v range: [-0.1754, 0.2821] -> mid: 0.05335, half_range: 0.22875
    
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    xn = (x - x_mid) / x_half
    vn = (v - v_mid) / v_half
    
    # Compute terms
    # Linear
    t1 = xn
    t2 = vn
    
    # Quadratic
    t3 = xn * xn
    t4 = vn * vn
    
    # Cubic
    t5 = xn * xn * xn
    t6 = vn * vn * vn
    
    # Cross
    t7 = xn * vn
    
    # Tanh saturation
    t8 = np.tanh(xn)
    t9 = np.tanh(vn)
    
    # Bias
    t0 = np.ones_like(xn)
    
    return (p0 * t0 +
            p1 * t1 +
            p2 * t2 +
            p3 * t3 +
            p4 * t4 +
            p5 * t5 +
            p6 * t6 +
            p7 * t7 +
            p8 * t8 +
            p9 * t9)
