import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rational (Padé-like) function.
    
    The model is of the form:
    a = (p0 + p1*X + p2*V + p3*X^2 + p4*X*V + p5*V^2) / (1 + p6*X + p7*V + p8*X^2 + p9*V^2)
    
    Inputs are scaled to improve numerical conditioning for the optimizer.
    """
    # Scale inputs to roughly [-1, 1] range to help BFGS
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    # Using midpoints and half-ranges for better centering
    x_scale = 0.55  # approx half-range of x
    v_scale = 0.25  # approx half-range of v
    
    X = x / x_scale
    V = v / v_scale
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator terms
    num_0 = p0
    num_1 = p1 * X
    num_2 = p2 * V
    num_3 = p3 * X**2
    num_4 = p4 * X * V
    num_5 = p5 * V**2
    numerator = num_0 + num_1 + num_2 + num_3 + num_4 + num_5
    
    # Denominator terms (1 + ...)
    den_0 = 1.0
    den_1 = p6 * X
    den_2 = p7 * V
    den_3 = p8 * X**2
    den_4 = p9 * V**2
    denominator = den_0 + den_1 + den_2 + den_3 + den_4
    
    # Avoid division by zero (though denominator should be close to 1 initially)
    # Add a small epsilon to prevent numerical issues if BFGS goes astray
    eps = 1e-8
    safe_denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    
    return numerator / safe_denominator
