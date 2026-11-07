import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a compact, well-scaled basis of
    physics-motivated nonlinearities: odd polynomials for restoring force,
    linear and cubic terms for damping, and cross-coupling terms.
    All features are scaled to O(1) magnitude in the training domain to
    avoid ill-conditioning during BFGS fitting.
    """
    p = params
    
    # Scaled inputs (training ranges: x in [-0.57, 0.53], v in [-0.18, 0.28])
    # Scale so typical values are O(1)
    xs = 2.0 * x       # ~ O(1)
    vs = 4.0 * v       # ~ O(1)
    
    # Restoring force basis (odd in x, for position-dependent acceleration)
    f0 = xs                     # linear restoring
    f1 = xs**3                  # cubic restoring (Duffing-like)
    f2 = xs**5                  # quintic restoring (softening/hardening)
    
    # Damping basis (odd in v, for velocity-dependent acceleration)
    f3 = vs                     # linear damping
    f4 = vs**3                  # cubic damping
    
    # Cross-coupling terms
    f5 = xs * vs                # linear coupling
    f6 = xs**2 * vs             # quadratic-x coupling with v
    f7 = xs * vs**3             # x with cubic v
    f8 = xs**3 * vs             # cubic x with v
    
    # Constant offset
    f9 = np.ones_like(x)
    
    # Combine with fitted parameters
    return (p[0]*f0 + p[1]*f1 + p[2]*f2 +
            p[3]*f3 + p[4]*f4 +
            p[5]*f5 + p[6]*f6 + p[7]*f7 + p[8]*f8 +
            p[9]*f9)
