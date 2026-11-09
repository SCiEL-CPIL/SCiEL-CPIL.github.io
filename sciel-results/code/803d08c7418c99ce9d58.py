"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for a nonlinear oscillator.
    
    The equation models the acceleration (dv/dt) of a nonlinear oscillator.
    We use a structure that captures:
    - Linear damping: c * v
    - Linear restoring force: k * x
    - Nonlinear stiffness: k3 * x^3 (Duffing-like)
    - Velocity nonlinearity: kv3 * v^3
    - Cross terms: kxv * x * v
    - Higher order interactions to capture complex dynamics
    
    Structure:
    y = p0 * x + p1 * v + p2 * x^2 + p3 * v^2 + p4 * x^3 
      + p5 * v^3 + p6 * x*v + p7 * x^2*v + p8 * x*v^2 + p9
    
    This is a cubic polynomial in x and v, which should be sufficient to
    capture the dynamics of a nonlinear oscillator. The time dependence
    is assumed to be captured through the state variables x and v,
    or the system is autonomous (no explicit t dependence).
    
    However, since t is an input, let's include some mild t-dependence
    or assume the system is autonomous and t is just a label.
    Given the ranges, t goes from 30 to 50, which might be a time index.
    If the system is autonomous, t shouldn't appear. But let's include
    a small t-dependent modulation to be safe, or just ignore t.
    
    Actually, for a standard oscillator, the acceleration depends on x and v,
    not explicitly on t (autonomous system). But if there's external forcing
    or parameter variation with time, t might matter.
    
    Let's try a model that primarily depends on x and v, with optional
    weak t-dependence. But we only have 10 params. Let's use them all
    for the x-v polynomial.
    
    If the system is truly autonomous, t is irrelevant. Let's first try
    ignoring t and using a cubic polynomial in x and v.
    
    Basis functions:
    1. 1 (constant)
    2. x
    3. v
    4. x^2
    5. v^2
    6. x^3
    7. v^3
    8. x*v
    9. x^2*v
    10. x*v^2
    
    This gives a flexible model for the acceleration.
    """
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Linear combination
    y = (p0 * x + p1 * v + p2 * x2 + p3 * v2 + p4 * x3 
         + p5 * v3 + p6 * xv + p7 * x2v + p8 * xv2 + p9)
    
    return y

# EVOLVE-BLOCK-END
