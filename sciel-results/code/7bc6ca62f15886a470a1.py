"""Symbolic regression with adaptive Fourier + Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Fourier series with Chebyshev polynomial trend.
    
    Parameters (10 total):
      p0: constant offset
      p1: linear coefficient (Chebyshev T1)
      p2: quadratic coefficient (Chebyshev T2)
      p3: cubic coefficient (Chebyshev T3)
      p4: amplitude of first Fourier mode
      p5: frequency of first Fourier mode (learnable!)
      p6: amplitude of second Fourier mode
      p7: frequency of second Fourier mode (learnable!)
      p8: phase shift for first mode
      p9: phase shift for second mode
    
    The learnable frequencies allow the model to adapt to the actual
    periodicity of the target function. Chebyshev polynomials provide
    better numerical conditioning than raw powers.
    """
    # Time guard
    if time.time() > _DEADLINE:
        return params[0] + params[1] * x

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Ensure we have at least 10 params
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
    p = p[:10]

    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    a1, f1, a2, f2, ph1, ph2 = p[4], p[5], p[6], p[7], p[8], p[9]

    # Clamp frequencies to avoid numerical issues
    f1_clamped = np.clip(f1, -20.0, 20.0)
    f2_clamped = np.clip(f2, -20.0, 20.0)

    # Chebyshev polynomials (normalized to [-1,1] domain)
    # x is in [-2, 2], so map to [-1, 1]: t = x/2
    t = x / 2.0
    T0 = np.ones_like(x)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t

    # Fourier terms with learnable frequencies and phases
    sin1 = np.sin(f1_clamped * x + ph1)
    cos1 = np.cos(f1_clamped * x + ph1)
    sin2 = np.sin(f2_clamped * x + ph2)
    cos2 = np.cos(f2_clamped * x + ph2)

    # Combine: polynomial trend + two adaptive Fourier modes
    # Use both sin and cos for each frequency to allow arbitrary phase
    y = (p0 * T0 +
         p1 * T1 +
         p2 * T2 +
         p3 * T3 +
         a1 * sin1 +
         a2 * sin2 +
         (p1 * 0.1) * cos1 +   # small coupling to keep optimization smooth
         (p2 * 0.1) * cos2)    # small coupling

    # Ensure finite output
    if not np.all(np.isfinite(y)):
        # Fallback: just polynomial part
        y = p0 * T0 + p1 * T1 + p2 * T2 + p3 * T3
        if not np.all(np.isfinite(y)):
            return np.zeros_like(x)

    return y
# EVOLVE-BLOCK-END
