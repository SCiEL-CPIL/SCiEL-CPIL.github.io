"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model for nonlinear oscillator acceleration.
    Uses 8 Gaussian RBFs with fixed centers in (x, v) space plus a linear term.
    Inputs are scaled to O(1) for numerical stability.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale inputs to roughly [-1, 1] range for better conditioning
    # x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    xs = x / 0.25   # ~[-0.82, 0.70]
    vs = v / 0.35   # ~[-0.97, 1.04]

    # Fixed RBF centers in scaled (x, v) space
    # Spread centers to cover the input domain
    centers_x = np.array([-0.6, -0.2, 0.2, 0.6, -0.4, 0.0, 0.4, -0.1])
    centers_v = np.array([-0.5, 0.3, -0.3, 0.5, 0.1, -0.1, 0.7, 0.0])

    # Gaussian RBF bandwidth
    sigma = 0.35

    # Compute squared distances from each point to each center
    dx = xs[:, np.newaxis] - centers_x[np.newaxis, :]
    dv = vs[:, np.newaxis] - centers_v[np.newaxis, :]
    dist_sq = dx * dx + dv * dv

    # Compute RBF values
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma * sigma))

    # Linear combination of RBFs plus a constant offset
    coeffs = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    rbf_contrib = rbf_vals @ coeffs

    # Add a small linear correction term using p9 and p0
    linear_term = p9 * xs + p0 * vs

    return rbf_contrib + linear_term
# EVOLVE-BLOCK-END
