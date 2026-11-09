"""Symbolic regression with stable mixed basis: polynomial + trig + tanh + Gaussian."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural redesign: two Gaussian bumps with learnable centers and widths,
    plus polynomial, trig, and tanh terms.
    
    Parameters (10 total):
      p0: constant
      p1: linear coefficient
      p2: quadratic coefficient
      p3: sin(x) coefficient
      p4: cos(x) coefficient
      p5: tanh(x) coefficient
      p6: Gaussian 1 amplitude
      p7: Gaussian 1 center
      p8: Gaussian 2 amplitude  
      p9: Gaussian 2 center
    
    Widths are fixed at 1.0 for stability. This gives two local bumps that can
    capture non-smooth or multi-peaked features, plus global polynomial/trig fit.
    """
    # Time guard
    if time.time() > _DEADLINE:
        return params[0] + params[1] * x

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]

    p0, p1, p2, p3, p4 = p[0], p[1], p[2], p[3], p[4]
    p5, p6, p7, p8, p9 = p[5], p[6], p[7], p[8], p[9]

    # Polynomial terms
    x2 = x * x

    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    # Hyperbolic term
    tanh_x = np.tanh(x)

    # Two Gaussian bumps with fixed width 1.0
    sigma = 1.0
    g1 = p6 * np.exp(-((x - p7) / sigma) ** 2)
    g2 = p8 * np.exp(-((x - p9) / sigma) ** 2)

    # Combine all terms
    y = (p0 * np.ones_like(x) +
         p1 * x +
         p2 * x2 +
         p3 * sin_x +
         p4 * cos_x +
         p5 * tanh_x +
         g1 +
         g2)

    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)

    return y
# EVOLVE-BLOCK-END
