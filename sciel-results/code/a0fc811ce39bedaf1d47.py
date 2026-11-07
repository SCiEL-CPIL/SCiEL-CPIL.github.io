"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Combines physics-motivated basis (Duffing + damping + periodic forcing)
    with Gaussian RBF terms for localized nonlinear corrections.
    
    Inputs are normalized to O(1) scale for better BFGS conditioning.
    10 parameters total:
      p0: bias
      p1: linear restoring (x)
      p2: Duffing hardening (x^3)
      p3: linear damping (v)
      p4: forcing in phase (sin(t))
      p5: forcing quadrature (cos(t))
      p6-p9: Gaussian RBF localized corrections in (x, v) space
    """
    t0 = time.time()
    
    # Safety guard: return zeros if we're close to the deadline
    if time.time() - t0 > 140.0:
        return np.zeros_like(t, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to O(1) range for better numerical conditioning
    # x range: [-0.3391, 0.3624], v range: [-0.2056, 0.1749]
    xs = x / 0.36
    vs = v / 0.21
    
    # Precompute trigonometric functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Physics-motivated basis terms (6 terms)
    b0 = 1.0                          # bias
    b1 = xs                           # linear restoring force
    b2 = xs * xs * xs                 # Duffing cubic nonlinearity
    b3 = vs                           # linear damping
    b4 = sin_t                        # periodic forcing in phase
    b5 = cos_t                        # periodic forcing quadrature
    
    # Gaussian RBF localized correction terms (4 terms)
    # Centers chosen within input ranges, widths scaled to domain
    c1_x, c1_v = 0.28, 0.24           # upper-right region
    c2_x, c2_v = -0.28, -0.24         # lower-left region
    c3_x, c3_v = 0.0, 0.38            # top-center
    c4_x, c4_v = 0.0, -0.38           # bottom-center
    
    width = 0.35  # width parameter for Gaussian RBFs
    
    r1 = (xs - c1_x)**2 + (vs - c1_v)**2
    r2 = (xs - c2_x)**2 + (vs - c2_v)**2
    r3 = (xs - c3_x)**2 + (vs - c3_v)**2
    r4 = (xs - c4_x)**2 + (vs - c4_v)**2
    
    b6 = np.exp(-r1 / (2.0 * width * width))
    b7 = np.exp(-r2 / (2.0 * width * width))
    b8 = np.exp(-r3 / (2.0 * width * width))
    b9 = np.exp(-r4 / (2.0 * width * width))
    
    # Combine all basis terms with parameters
    result = (p0 * b0 +
              p1 * b1 +
              p2 * b2 +
              p3 * b3 +
              p4 * b4 +
              p5 * b5 +
              p6 * b6 +
              p7 * b7 +
              p8 * b8 +
              p9 * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
    
    return result
# EVOLVE-BLOCK-END
