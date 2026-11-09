# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses a Duffing-van der Pol-inspired structure with Chebyshev basis functions.
    The Chebyshev polynomials provide a well-conditioned basis over the bounded
    training interval, which helps BFGS converge reliably from an initial guess
    of all ones. The structure separates position-dependent (stiffness) and
    velocity-dependent (damping) terms with cross-coupling.
    
    Basis terms (10 parameters):
    p[0]: T1_x  (linear position)
    p[1]: T3_x  (cubic position - Duffing nonlinearity)
    p[2]: T2_x  (quadratic position - asymmetry)
    p[3]: T1_v  (linear velocity - linear damping)
    p[4]: T3_v  (cubic velocity - nonlinear damping)
    p[5]: T2_v  (quadratic velocity)
    p[6]: T1_x * T1_v  (cross coupling)
    p[7]: T2_x * T1_v  (quadratic position, linear velocity)
    p[8]: T1_x * T2_v  (linear position, quadratic velocity)
    p[9]: 1           (bias/constant)
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Standardize inputs to [-1, 1] based on training ranges
    # x range: [-0.5728, 0.5275]
    # v range: [-0.1754, 0.2821]
    
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_center = (x_min + x_max) / 2.0
    x_radius = (x_max - x_min) / 2.0
    
    v_center = (v_min + v_max) / 2.0
    v_radius = (v_max - v_min) / 2.0
    
    X = (x - x_center) / x_radius
    V = (v - v_center) / v_radius
    
    # Clip to [-1, 1] for numerical stability with Chebyshev polynomials
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_n
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    
    T1_x = X
    T2_x = 2.0 * X * X - 1.0
    T3_x = 4.0 * X * X * X - 3.0 * X
    
    T1_v = V
    T2_v = 2.0 * V * V - 1.0
    T3_v = 4.0 * V * V * V - 3.0 * V
    
    # Construct the linear combination
    a = (p[0] * T1_x +
         p[1] * T3_x +
         p[2] * T2_x +
         p[3] * T1_v +
         p[4] * T3_v +
         p[5] * T2_v +
         p[6] * T1_x * T1_v +
         p[7] * T2_x * T1_v +
         p[8] * T1_x * T2_v +
         p[9])
    
    return a
# EVOLVE-BLOCK-END
