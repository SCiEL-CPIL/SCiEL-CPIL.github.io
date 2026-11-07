"""Autonomous nonlinear oscillator model (Duffing-Van der Pol) without explicit drive."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a self-sustained nonlinear oscillator.
    
    Structural change: Remove explicit sin(t)/cos(t) drive terms.
    The oscillator is modeled as autonomous (limit-cycle dynamics),
    with all 10 parameters fitting the intrinsic nonlinear dynamics.
    
    Basis functions (scaled inputs xn = x*2, vn = v*2):
      p0 * xn          (linear restoring force)
      p1 * xn^3        (Duffing cubic stiffness)
      p2 * xn^2        (quadratic position)
      p3 * vn          (linear damping)
      p4 * vn^3        (cubic velocity saturation)
      p5 * vn^2        (quadratic velocity / energy term)
      p6 * xn*vn       (cross-coupling)
      p7 * xn^2*vn     (Van der Pol negative damping)
      p8 * xn^2*vn^2   (higher-order position-velocity coupling)
      p9               (constant offset)
    
    This pure autonomous model lets BFGS fit the limit-cycle structure
    directly without competing with drive-term coefficients.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to O(1) for better BFGS conditioning
    # x in [-0.2056, 0.1749] -> xn in [-0.41, 0.35]
    # v in [-0.3391, 0.3624] -> vn in [-0.68, 0.72]
    xn = x * 2.0
    vn = v * 2.0
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Construct autonomous model
    a = p0 * xn
    a = a + p1 * xn3
    a = a + p2 * xn2
    a = a + p3 * vn
    a = a + p4 * vn3
    a = a + p5 * vn2
    a = a + p6 * xn * vn
    a = a + p7 * xn2 * vn
    a = a + p8 * xn2 * vn2
    a = a + p9
    
    return a
# EVOLVE-BLOCK-END
