import numpy as np
import time

# Domain: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structured Chebyshev polynomial expansion for a nonlinear oscillator.
    Uses a basis tailored to Duffing/Van der Pol dynamics:
    - Duffing: x'' + delta*x' + alpha*x + beta*x^3 = 0  => x' = -delta*x - alpha*x - beta*x^3 (implicit)
    - Van der Pol: x'' + mu*(x^2-1)*x' + x = 0  => x' = -x - mu*(x^2-1)*x' (implicit)
    
    For a first-order system or reduced form, we model the target as a function of (x, v).
    The basis includes:
    1. Constant
    2. x
    3. v
    4. x^2
    5. v^2
    6. x*v
    7. x^3 (Duffing nonlinearity)
    8. v^3 (cubic damping/restoring)
    9. x^2*v (interaction)
    10. x*v^2 (interaction)
    
    This is a monomial basis up to degree 3, which is more directly aligned with
    common nonlinear oscillator forms than pure Chebyshev.
    """
    # Normalize inputs to [-1, 1] for better conditioning
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale

    # Monomial basis
    # 0: 1
    # 1: xn
    # 2: vn
    # 3: xn^2
    # 4: vn^2
    # 5: xn * vn
    # 6: xn^3
    # 7: vn^3
    # 8: xn^2 * vn
    # 9: xn * vn^2

    term0 = np.ones_like(xn)
    term1 = xn
    term2 = vn
    term3 = xn * xn
    term4 = vn * vn
    term5 = xn * vn
    term6 = xn * xn * xn
    term7 = vn * vn * vn
    term8 = term3 * vn
    term9 = term1 * term4

    # Combine with parameters
    result = (params[0] * term0 +
              params[1] * term1 +
              params[2] * term2 +
              params[3] * term3 +
              params[4] * term4 +
              params[5] * term5 +
              params[6] * term6 +
              params[7] * term7 +
              params[8] * term8 +
              params[9] * term9)

    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)

    return result
