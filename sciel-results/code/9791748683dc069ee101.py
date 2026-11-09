"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 150.0  # seconds


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structurally different model: a generalized logistic (sigmoid) response
    plus a low-order polynomial.
    
    The generalized logistic form:
        L(x) = A / (1 + exp(-B * (x - C)))
    captures S-shaped transitions, saturation, and step-like behaviors that
    are common in many natural and synthetic functions. Adding a quadratic
    polynomial allows for curvature that the sigmoid alone cannot capture.
    
    Parameters (length 10, but only first 5 are used meaningfully; rest are
    included to satisfy the interface):
      p[0] = A: amplitude / asymptotic value
      p[1] = B: steepness of the transition
      p[2] = C: center / midpoint of the transition
      p[3] = quadratic coefficient (x^2 term)
      p[4] = linear coefficient (x term)
      p[5]..p[9] = reserved (set to 0 in effect)
    
    The model is:
        y = A / (1 + exp(-B*(x - C))) + p[3]*x^2 + p[4]*x
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety fallback: simple linear
        p = np.asarray(params, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        result = p[4] * x_arr + p[0]
        return np.where(np.isfinite(result), result, 0.0)
    
    p = np.asarray(params, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    
    # Clip parameters to prevent numerical overflow in exp
    A = p[0]
    B = np.clip(p[1], -20.0, 20.0)
    C = np.clip(p[2], -10.0, 10.0)
    quad_coeff = p[3]
    lin_coeff = p[4]
    
    # Generalized logistic (sigmoid) term
    # Use a numerically stable version of the sigmoid
    z = -B * (x_arr - C)
    # For large positive z, exp(-z) is small; for large negative z, exp(z) is small
    # Use the stable form: sigmoid(z) = 1 / (1 + exp(-z))
    # But here we have 1/(1+exp(z)) effectively when we write A/(1+exp(-B*(x-C)))
    # Let's compute exp(z) safely
    # z can be in range roughly [-20*6, 20*6] = [-120, 120] after clipping B
    # exp(120) overflows, so we need to be careful
    
    # Stable computation of 1/(1 + exp(z)):
    # If z > 0: 1/(1+exp(z)) = exp(-z)/(1+exp(-z))
    # If z <= 0: 1/(1+exp(z))
    sigmoid_val = np.where(
        z > 0,
        np.exp(-z) / (1.0 + np.exp(-z)),
        1.0 / (1.0 + np.exp(z))
    )
    
    logistic_part = A * sigmoid_val
    
    # Polynomial part
    poly_part = quad_coeff * x_arr**2 + lin_coeff * x_arr
    
    result = logistic_part + poly_part
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
