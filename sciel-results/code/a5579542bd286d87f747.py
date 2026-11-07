"""Symbolic regression: nonlinear oscillator with parametric excitation."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

# Time budget guard
_T0 = time.time()
_TIME_LIMIT = 140.0  # Leave 10s safety margin


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed nonlinear oscillator model with parametric excitation.
    
    Model: a = c1*x + c2*x^3 + c3*v + c4*x*v + c5*x^2 + c6*v^2 
             + c7*sin(t)*x + c8*cos(t)*x + c9*sin(t)*v + c10
    
    This captures:
    - Linear restoring force (c1*x)
    - Cubic nonlinearity (c2*x^3) - Duffing-like
    - Linear damping (c3*v)
    - Velocity-position coupling (c4*x*v)
    - Quadratic position term (c5*x^2)
    - Quadratic velocity term (c6*v^2)
    - Parametric excitation in stiffness (c7*sin(t)*x, c8*cos(t)*x)
    - Parametric excitation in damping (c9*sin(t)*v)
    - Constant offset (c10)
    
    params[0:10] = coefficients c1 through c10
    """
    # Check time budget - if close to limit, return a simple fallback
    elapsed = time.time() - _T0
    if elapsed > _TIME_LIMIT:
        # Fallback: simple linear model
        return params[0] * x + params[1] * v + params[2]
    
    p = params  # shape (10,)
    
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = p
    
    # Precompute trig terms (vectorized, efficient)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Core oscillator terms
    linear_restore = c1 * x
    cubic_nonlinear = c2 * x**3
    linear_damping = c3 * v
    cross_term = c4 * x * v
    quad_pos = c5 * x**2
    quad_vel = c6 * v**2
    
    # Parametric excitation terms
    param_excite_pos = c7 * sin_t * x + c8 * cos_t * x
    param_excite_vel = c9 * sin_t * v
    
    # Constant offset
    offset = c10
    
    return (linear_restore + cubic_nonlinear + linear_damping + cross_term 
            + quad_pos + quad_vel + param_excite_pos + param_excite_vel + offset)
# EVOLVE-BLOCK-END
