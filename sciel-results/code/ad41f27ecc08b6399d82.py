"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for a nonlinear oscillator.
    
    The acceleration is modeled as:
        a = -dV/dx - gamma(v) * v
    
    where V(x) is a polynomial potential and gamma(v) is a bounded damping function.
    
    This structure captures conservative forces from a potential and
    velocity-dependent damping, which is physically natural for oscillators.
    
    Parameters (10 total):
        p[0]: linear stiffness (k1)
        p[1]: cubic stiffness (k3)
        p[2]: quintic stiffness (k5)
        p[3]: linear damping (c1)
        p[4]: cubic damping (c3)
        p[5]: damping scale (ds)
        p[6]: x-v coupling coefficient (cv)
        p[7]: v^2 coupling coefficient (cv2)
        p[8]: bias term
        p[9]: additional linear v term
    
    Inputs:
      x: position
      v: velocity
      params: array of 10 coefficients
      
    Returns:
      a: acceleration
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Extract parameters
    k1 = p[0]
    k3 = p[1]
    k5 = p[2]
    c1 = p[3]
    c3 = p[4]
    ds = p[5]
    cv = p[6]
    cv2 = p[7]
    bias = p[8]
    lv = p[9]
    
    # Ensure damping scale is positive for stability
    ds = np.maximum(np.abs(ds), 1e-6)
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    v2 = v * v
    v3 = v2 * v
    
    # Potential gradient: dV/dx = k1*x + k3*x^3 + k5*x^5
    # Conservative force: -dV/dx
    force_conservative = -(k1 * x + k3 * x3 + k5 * x5)
    
    # Damping term: gamma(v) * v where gamma(v) = c1 + c3 * tanh(v^2 / ds)
    # tanh keeps gamma bounded
    tanh_arg = v2 / ds
    gamma = c1 + c3 * np.tanh(tanh_arg)
    damping_force = -gamma * v
    
    # Cross terms: cv * x * v + cv2 * v^2
    cross_force = cv * x * v + cv2 * v2
    
    # Additional linear velocity term
    linear_v = lv * v
    
    # Combine all terms
    a = force_conservative + damping_force + cross_force + linear_v + bias
    
    return a
# EVOLVE-BLOCK-END
