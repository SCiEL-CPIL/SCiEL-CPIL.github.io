"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
START_TIME = time.time()
DEADLINE = 120.0  # 120s limit, keep margin

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a 2D Chebyshev polynomial basis.
    
    Chebyshev polynomials provide an optimal basis for approximating smooth functions
    on bounded intervals, leading to better-conditioned systems for BFGS optimization.
    
    We scale x and v to [-1, 1] based on the training ranges:
    x range: [-0.5728, 0.5275] -> center = -0.02265, half_range = 0.55015
    v range: [-0.1754, 0.2821] -> center = 0.05335, half_range = 0.22875
    
    Basis: T_i(x_scaled) * T_j(v_scaled) for i+j <= 4 (total 15 terms, but we use 10)
    We select a subset of 10 terms that covers the most important low-order interactions.
    
    The 10 terms (i, j) where i is degree in x, j is degree in v:
    (0,0), (1,0), (0,1), (2,0), (0,2), (3,0), (1,1), (0,3), (4,0), (2,1)
    
    This covers:
    - Constant
    - Linear x, Linear v
    - Quadratic x^2, v^2
    - Cubic x^3, v^3, x*v
    - Quartic x^4, x^2*v
    """
    # Check deadline
    if time.time() > DEADLINE:
        return np.zeros_like(x)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale x and v to [-1, 1]
    x_center = -0.02265
    x_half_range = 0.55015
    v_center = 0.05335
    v_half_range = 0.22875
    
    x_s = (x - x_center) / x_half_range
    v_s = (v - v_center) / v_half_range
    
    # Clip to [-1, 1] to avoid numerical issues outside training range
    x_s = np.clip(x_s, -1.0, 1.0)
    v_s = np.clip(v_s, -1.0, 1.0)
    
    # Compute Chebyshev polynomials up to degree 4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    
    # For x
    Tx0 = np.ones_like(x_s)
    Tx1 = x_s
    Tx2 = 2 * x_s * x_s - 1
    Tx3 = 4 * x_s * x_s * x_s - 3 * x_s
    Tx4 = 8 * x_s * x_s * x_s * x_s - 8 * x_s * x_s + 1
    
    # For v
    Tv0 = np.ones_like(v_s)
    Tv1 = v_s
    Tv2 = 2 * v_s * v_s - 1
    Tv3 = 4 * v_s * v_s * v_s - 3 * v_s
    Tv4 = 8 * v_s * v_s * v_s * v_s - 8 * v_s * v_s + 1
    
    # Combine 10 terms:
    # p0 * T0(x) * T0(v) = p0
    # p1 * T1(x) * T0(v) = p1 * x_s
    # p2 * T0(x) * T1(v) = p2 * v_s
    # p3 * T2(x) * T0(v) = p3 * Tx2
    # p4 * T0(x) * T2(v) = p4 * Tv2
    # p5 * T3(x) * T0(v) = p5 * Tx3
    # p6 * T1(x) * T1(v) = p6 * x_s * v_s
    # p7 * T0(x) * T3(v) = p7 * Tv3
    # p8 * T4(x) * T0(v) = p8 * Tx4
    # p9 * T2(x) * T1(v) = p9 * Tx2 * v_s
    
    return (p0 * Tx0 * Tv0 + 
            p1 * Tx1 * Tv0 + 
            p2 * Tx0 * Tv1 + 
            p3 * Tx2 * Tv0 + 
            p4 * Tx0 * Tv2 + 
            p5 * Tx3 * Tv0 + 
            p6 * Tx1 * Tv1 + 
            p7 * Tx0 * Tv3 + 
            p8 * Tx4 * Tv0 + 
            p9 * Tx2 * Tv1)
# EVOLVE-BLOCK-END
