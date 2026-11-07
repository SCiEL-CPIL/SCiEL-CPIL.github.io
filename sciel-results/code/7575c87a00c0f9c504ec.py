import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Generalized Chebyshev basis expansion for nonlinear oscillator dynamics.
    
    This model uses Chebyshev polynomials of the first kind on the normalized
    input domain to provide a well-conditioned, flexible basis for approximating
    the acceleration function a = f(x, v).
    
    The inputs x and v are normalized to [-1, 1] based on the known training ranges:
    x in [-0.5728, 0.5275] -> center -0.02265, half-range 0.55015
    v in [-0.1754, 0.2821] -> center 0.05335, half-range 0.22875
    
    Basis functions (10 parameters):
    p0: Constant (T0(x)*T0(v))
    p1: Linear x (T1(x)*T0(v))
    p2: Linear v (T0(x)*T1(v))
    p3: Quadratic x (T2(x)*T0(v))
    p4: Quadratic v (T0(x)*T2(v))
    p5: Interaction x*v (T1(x)*T1(v))
    p6: Cubic x (T3(x)*T0(v))
    p7: Cubic v (T0(x)*T3(v))
    p8: Mixed x^2*v (T2(x)*T1(v))
    p9: Mixed x*v^2 (T1(x)*T2(v))
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to [-1, 1] for Chebyshev stability
    # x range: [-0.5728, 0.5275]
    x_min, x_max = -0.5728, 0.5275
    x_center = (x_min + x_max) / 2.0
    x_range = (x_max - x_min) / 2.0
    x_norm = (x - x_center) / x_range
    
    # v range: [-0.1754, 0.2821]
    v_min, v_max = -0.1754, 0.2821
    v_center = (v_min + v_max) / 2.0
    v_range = (v_max - v_min) / 2.0
    v_norm = (v - v_center) / v_range
    
    # Clip to [-1, 1] to handle any slight out-of-distribution noise robustly
    # This prevents Chebyshev polynomials from blowing up if test data is slightly outside
    x_norm = np.clip(x_norm, -1.0, 1.0)
    v_norm = np.clip(v_norm, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_n(t) = cos(n * arccos(t))
    # Using recurrence for efficiency and stability:
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2t T_n(t) - T_{n-1}(t)
    
    T0_x = np.ones_like(x_norm)
    T1_x = x_norm
    T2_x = 2.0 * x_norm * T1_x - T0_x
    T3_x = 2.0 * x_norm * T2_x - T1_x
    
    T0_v = np.ones_like(v_norm)
    T1_v = v_norm
    T2_v = 2.0 * v_norm * T1_v - T0_v
    T3_v = 2.0 * v_norm * T2_v - T1_v
    
    # Construct basis terms
    b0 = T0_x * T0_v
    b1 = T1_x * T0_v
    b2 = T0_x * T1_v
    b3 = T2_x * T0_v
    b4 = T0_x * T2_v
    b5 = T1_x * T1_v
    b6 = T3_x * T0_v
    b7 = T0_x * T3_v
    b8 = T2_x * T1_v
    b9 = T1_x * T2_v
    
    # Linear combination
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return result
