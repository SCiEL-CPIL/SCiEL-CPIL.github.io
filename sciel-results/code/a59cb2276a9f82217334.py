import numpy as np

# Scales for normalization to improve conditioning of the basis functions
# x range approx [-0.57, 0.53] -> center ~ -0.02, scale ~ 0.55
# v range approx [-0.18, 0.28] -> center ~ 0.05, scale ~ 0.25
x_scale = 0.55
v_scale = 0.25
x_center = -0.02
v_center = 0.05

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis for a nonlinear oscillator (Duffing-van der Pol style).
    Uses normalized inputs to ensure well-conditioned parameters for BFGS.
    """
    # Normalize inputs
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale

    # Compute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn

    # Define 10 basis terms
    # 0: Constant
    t0 = np.ones_like(xn)
    # 1: Linear x
    t1 = xn
    # 2: Linear v
    t2 = vn
    # 3: Cubic x (Duffing stiffness)
    t3 = xn3
    # 4: Cubic v (Nonlinear damping)
    t4 = vn3
    # 5: Quadratic x
    t5 = xn2
    # 6: Quadratic v
    t6 = vn2
    # 7: Cross x*v
    t7 = xn * vn
    # 8: Cross x^2*v
    t8 = xn2 * vn
    # 9: Cross x*v^2
    t9 = xn * vn2

    # Linear combination
    result = (params[0] * t0 +
              params[1] * t1 +
              params[2] * t2 +
              params[3] * t3 +
              params[4] * t4 +
              params[5] * t5 +
              params[6] * t6 +
              params[7] * t7 +
              params[8] * t8 +
              params[9] * t9)

    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)

    return result
