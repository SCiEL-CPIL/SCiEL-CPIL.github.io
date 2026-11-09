"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    # Nonlinear parametric model: parameters appear inside transcendental functions,
    # allowing BFGS to discover functional form (frequency, phase, decay, etc.)
    # rather than just fitting linear combinations of fixed bases.
    #
    # Model: y = a*sin(w*x + p) + b*cos(w*x + p) + c*exp(-d*x^2) + e*x^2 + f*x + g
    #        + h*tanh(s*x) + i*x^3
    #
    # Params mapping:
    #   p0 = a (sine amplitude)
    #   p1 = b (cosine amplitude)
    #   p2 = w (angular frequency, learnable)
    #   p3 = p (phase shift, learnable)
    #   p4 = c (gaussian amplitude)
    #   p5 = d (gaussian decay rate, learnable)
    #   p6 = e (quadratic coeff)
    #   p7 = f (linear coeff)
    #   p8 = g (constant)
    #   p9 = h (tanh amplitude)
    
    a = params[0]
    b = params[1]
    w = params[2]
    p = params[3]
    c = params[4]
    d = params[5]
    e = params[6]
    f = params[7]
    g = params[8]
    h = params[9]
    
    # Clamp d to avoid exp overflow (d*x^2 can be large for negative d)
    d_safe = np.clip(d, -10.0, 10.0)
    
    y = (a * np.sin(w * x + p) +
         b * np.cos(w * x + p) +
         c * np.exp(-d_safe * x**2) +
         e * x**2 +
         f * x +
         g +
         h * np.tanh(x) +
         e * x**3 / 10.0)  # small cubic term for flexibility
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
