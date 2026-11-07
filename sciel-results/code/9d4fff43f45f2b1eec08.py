"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a self-sustained nonlinear oscillator.
    Assumes autonomous dynamics (Van der Pol / Duffing type) with no explicit time dependence.
    Uses a full 3rd-order polynomial basis in position x and velocity v to capture
    limit cycle behavior through coupled restoring and damping nonlinearities.
    Uses up to 10 fitted constants.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute powers for efficiency and numerical stability
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v

    # 1. Constant offset
    term_const = p0

    # 2. Linear restoring force (Duffing linear part)
    term_x = p1 * x

    # 3. Linear damping (Van der Pol linear part)
    term_v = p2 * v

    # 4. Quadratic position term
    term_x2 = p3 * x2

    # 5. Cross term x*v
    term_xv = p4 * x * v

    # 6. Quadratic velocity term
    term_v2 = p5 * v2

    # 7. Cubic restoring force (Duffing nonlinear part)
    term_x3 = p6 * x3

    # 8. Van der Pol damping term (x^2 * v) - negative damping for small x, positive for large x
    term_x2v = p7 * x2 * v

    # 9. Mixed term x * v^2
    term_xv2 = p8 * x * v2

    # 10. Cubic damping (velocity saturation)
    term_v3 = p9 * v3

    return (term_const + term_x + term_v + term_x2 + term_xv +
            term_v2 + term_x3 + term_x2v + term_xv2 + term_v3)
# EVOLVE-BLOCK-END
