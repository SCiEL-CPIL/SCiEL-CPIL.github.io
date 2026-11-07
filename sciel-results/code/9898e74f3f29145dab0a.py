import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to improve conditioning for the optimizer
    # Based on training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    x_scale = 0.21
    v_scale = 0.37
    
    xn = x / x_scale
    vn = v / v_scale
    
    # Pre-compute powers of normalized variables
    xn2 = xn * xn
    xn3 = xn2 * xn
    xn5 = xn3 * xn2
    
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Basis functions
    # 1. Linear restoring
    f1 = xn
    # 2. Cubic restoring
    f2 = xn3
    # 3. Quintic restoring
    f3 = xn5
    # 4. Linear damping
    f4 = vn
    # 5. Cubic damping
    f5 = vn3
    # 6. Van der Pol style coupling (x^2 * v)
    f6 = xn2 * vn
    # 7. Cross term (x * v)
    f7 = xn * vn
    # 8. Quadratic restoring (asymmetry)
    f8 = xn2
    # 9. Quadratic damping (asymmetry)
    f9 = vn2
    # 10. Bias
    f10 = 1.0
    
    # Combine with parameters
    # params[0] to params[9]
    result = (
        params[0] * f1 +
        params[1] * f2 +
        params[2] * f3 +
        params[3] * f4 +
        params[4] * f5 +
        params[5] * f6 +
        params[6] * f7 +
        params[7] * f8 +
        params[8] * f9 +
        params[9] * f10
    )
    
    return result
