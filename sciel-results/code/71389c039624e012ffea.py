"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# The evaluator calls equation(t, x, v, params).
# Physics-informed nonlinear oscillator model:
#   a = -f(x) - g(x,v)*v + F(t)
# where f(x) is restoring force, g(x,v) is state-dependent damping, F(t) is forcing.
#
# Structure (10 params):
#   Restoring force:  f(x) = p0*x + p1*x^3 + p2*x^5
#   Damping:          g(x,v) = p3 + p4*x^2
#   Forcing:          F(t) = p5*sin(t) + p6*cos(t)
#   Cross/nonlinear:  p7*x*v + p8*v^3 + p9*x^2*v
#
# This directly encodes Duffing (x^3), Van der Pol (x^2*v), and
# higher-order nonlinearities, giving BFGS a well-structured landscape.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed closed-form expression for nonlinear oscillator acceleration.
    
    a = -(p0*x + p1*x^3 + p2*x^5) - (p3 + p4*x^2)*v + p5*sin(t) + p6*cos(t)
        + p7*x*v + p8*v^3 + p9*x^2*v
    
    This covers:
      - Linear oscillator: p0>0, others 0
      - Duffing: p1 != 0
      - Van der Pol: p9 != 0 (x^2*v term)
      - Cubic damping: p8 != 0
      - Periodic forcing: p5, p6 != 0
    """
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v3 = v2 * v
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Restoring force (negative sign: acceleration opposes displacement)
    restoring = -(params[0] * x + params[1] * x3 + params[2] * x5)
    
    # State-dependent damping (negative sign: opposes velocity)
    damping = -(params[3] + params[4] * x2) * v
    
    # Periodic forcing
    forcing = params[5] * sin_t + params[6] * cos_t
    
    # Additional nonlinear coupling terms
    coupling = params[7] * x * v + params[8] * v3 + params[9] * x2 * v
    
    y = restoring + damping + forcing + coupling
    
    return y
# EVOLVE-BLOCK-END
