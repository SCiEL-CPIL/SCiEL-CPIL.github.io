"""
Combined strategy: Apply input normalization to the rational function model.

The current program uses a rational function form which is expressive but suffers
from poor conditioning because the inputs x and v are not scaled. The BFGS optimizer
struggles with the disparate scales and offsets of x and v.

Inspiration 3 and observation data indicate that scaling inputs to a unit-like range
provides a massive improvement in optimizer convergence.

This program combines:
1. The structural flexibility of the rational restoring/damping forces from the current program.
2. The input normalization technique from Inspiration 1 and 3 to improve the condition number
   of the Hessian/Jacobian for BFGS.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Normalization constants based on training range
# x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts acceleration for a nonlinear oscillator using normalized rational function components.
    
    Model structure (applied to normalized inputs xn, vn):
    a = p0 
      + p1 * xn * (1 + p2*xn^2) / (1 + p3*xn^2)   # rational restoring force
      + p4 * vn * (1 + p5*vn^2) / (1 + p6*vn^2)   # rational damping force
      + p7 * xn * vn                              # linear coupling
      + p8 * xn^2 * vn                            # nonlinear coupling
      + p9 * vn * xn^2                            # additional nonlinear coupling
    
    Normalization ensures that the derivatives with respect to params are well-scaled,
    allowing BFGS to converge effectively.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Normalize inputs to roughly [-1, 1]
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    p = params
    
    # Ensure numerical stability: add small epsilon to denominators
    eps = 1e-8
    
    x2 = xn * xn
    v2 = vn * vn
    
    # Rational restoring force: xn * (1 + p2*xn^2) / (1 + p3*xn^2)
    denom_x = 1.0 + p[3] * x2
    # Safe division
    denom_x_safe = np.where(np.abs(denom_x) < eps, np.sign(denom_x) * eps + (denom_x == 0) * eps, denom_x)
    restoring = xn * (1.0 + p[2] * x2) / denom_x_safe
    
    # Rational damping force: vn * (1 + p5*vn^2) / (1 + p6*vn^2)
    denom_v = 1.0 + p[6] * v2
    # Safe division
    denom_v_safe = np.where(np.abs(denom_v) < eps, np.sign(denom_v) * eps + (denom_v == 0) * eps, denom_v)
    damping = vn * (1.0 + p[5] * v2) / denom_v_safe
    
    # Combine all terms
    result = (
        p[0]                                    # bias
        + p[1] * restoring                      # rational restoring force
        + p[4] * damping                        # rational damping force
        + p[7] * xn * vn                        # linear coupling
        + p[8] * x2 * vn                        # quadratic-in-x coupling
        + p[9] * vn * x2                        # another nonlinear coupling
    )
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
