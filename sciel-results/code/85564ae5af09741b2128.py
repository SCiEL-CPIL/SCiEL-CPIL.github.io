"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 145.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Adaptive Frequency Sinusoidal Model with Linear State Terms.

    The 10 parameters are:
      p0, p1, p2 : Amplitude, Frequency, Phase of 1st harmonic
      p3, p4, p5 : Amplitude, Frequency, Phase of 2nd harmonic
      p6         : Linear position coefficient
      p7         : Linear velocity coefficient
      p8         : Quadratic position coefficient
      p9         : Bias / Constant offset

    This model allows the periodic forcing frequencies to be fit to the data,
    unlike the fixed-frequency model in the parent.
    """
    # Safety margin guard
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(x, dtype=float) # Correction: v should be v
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    (a1, w1, ph1, a2, w2, ph2, bx, bv, bx2, bias) = p

    # Calculate sinusoidal terms
    # Note: BFGS might try negative frequencies. sin(-wt) = -sin(wt), so it's symmetric.
    # However, if w is very large, sin can oscillate rapidly.
    # To prevent numerical issues with extremely high frequencies, we could clip,
    # but let's allow BFGS to explore.
    
    term1 = a1 * np.sin(w1 * t + ph1)
    term2 = a2 * np.sin(w2 * t + ph2)
    
    # State terms
    state_term = bx * x + bv * v + bx2 * x * x
    
    out = term1 + term2 + state_term + bias
    
    # Guarantee finiteness and correct shape
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
# EVOLVE-BLOCK-END
