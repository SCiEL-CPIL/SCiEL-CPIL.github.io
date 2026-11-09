import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Duffing-van der Pol inspired structure.
    Combines physics-motivated nonlinear terms with flexible cross-interactions.
    """
    # Deadline guard - track elapsed time for safety
    start_time = time.time()
    
    # Scaling inputs to O(1) range for numerical stability
    # x in [-0.5728, 0.5275] -> scale by 0.6
    # v in [-0.1754, 0.2821] -> scale by 0.3
    xs = x / 0.6
    vs = v / 0.3
    
    # Precompute powers and products
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    x2v = xs2 * vs
    xv2 = xs * vs2
    x3v = xs3 * vs
    xvs = xs * vs
    
    # Physics-inspired nonlinear oscillator terms:
    # Duffing: nonlinear restoring force (cubic in x)
    # Van der Pol: nonlinear damping (x^2 * v)
    # Plus flexible cross terms for general nonlinear dynamics
    
    # Term 0: Linear restoring force (Hooke's law)
    t0 = xs
    # Term 1: Linear damping (viscous)
    t1 = vs
    # Term 2: Quadratic position (asymmetric stiffness)
    t2 = xs2
    # Term 3: Quadratic velocity (nonlinear drag)
    t3 = vs2
    # Term 4: Cross term (position-velocity coupling)
    t4 = xvs
    # Term 5: Cubic position (Duffing nonlinearity)
    t5 = xs3
    # Term 6: Cubic velocity (higher-order drag)
    t6 = vs3
    # Term 7: Van der Pol nonlinear damping
    t7 = x2v
    # Term 8: Higher-order cross (x*v^2)
    t8 = xv2
    # Term 9: Cubic-velocity cross (x^3*v) for strong nonlinear coupling
    t9 = x3v
    
    # Linear combination with fitted parameters
    result = (params[0] * t0 +
              params[1] * t1 +
              params[2] * t2 +
              params[3] * t3 +
              params[4] * t4 +
              params[5] * t5 +
              params[6] * t6 +
              params[7] * t7 +
              params[8] * t8 +
              params[9] * t9)
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
