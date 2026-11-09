# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_features(t, x, v):
    """Build a rich set of nonlinear features for the oscillator."""
    # Basic terms
    ones = np.ones_like(t)
    t_lin = t
    x_lin = x
    v_lin = v
    
    # Polynomial terms in x and v
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x4 = x2 * x2
    v4 = v2 * v2
    
    # Cross terms
    x_v = x * v
    x_v2 = x * v2
    x2_v = x2 * v
    x_v3 = x * v3
    x2_v2 = x2 * v2
    
    # Time-dependent driving terms (harmonic)
    # Since we don't know the exact driving frequency, we include
    # a few candidate harmonics with fitted amplitudes
    # Use a range of frequencies that might be present
    # But to keep it simple and within 10 params, we'll use:
    # t itself, and maybe sin/cos of t with some frequency
    # Actually, let's think about this differently.
    
    # The task says "time-dependent driving force"
    # Common forms: A*cos(w*t + phi) or A*sin(w*t)
    # Since we have 10 params total, let's allocate:
    # - 1 for constant
    # - 1 for linear t (or driving)
    # - 1 for linear x (spring)
    # - 1 for cubic x (nonlinear spring)
    # - 1 for linear v (damping)
    # - 1 for cubic v (nonlinear damping)
    # - 1 for x*v (cross damping)
    # - 1 for x^2*v or x*v^2
    # - 2 for driving: amplitude and phase of cos/sin
    
    # Let's use sin(t) and cos(t) as driving terms with fitted amplitudes
    # This assumes driving frequency is 1.0, which may not be correct.
    # Alternative: use t directly as a driving term.
    
    # Given the input ranges: t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # t is large, so linear t might dominate. Let's include it.
    
    # For driving, let's try: A*cos(w*t) + B*sin(w*t) but we don't know w.
    # Let's just use t, sin(t), cos(t) and let the fitter decide.
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Higher order time terms might not be needed
    t2 = t * t
    
    # Let's select 10 features:
    features = np.column_stack([
        ones,      # p0: constant
        t,         # p1: linear time (driving?)
        x,         # p2: linear spring
        x3,        # p3: cubic spring (nonlinearity)
        v,         # p4: linear damping
        v3,        # p5: cubic damping (nonlinearity)
        x_v,       # p6: cross damping
        x2_v,      # p7: quadratic spring * velocity
        sin_t,     # p8: sinusoidal driving
        cos_t,     # p9: cosinusoidal driving
    ])
    
    return features


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form equation for a damped nonlinear oscillator with time-dependent driving.
    
    Uses a linear combination of 10 nonlinear features.
    The evaluator fits params using BFGS from initial value of all ones.
    """
    features = _build_features(t, x, v)
    return features @ params
# EVOLVE-BLOCK-END
