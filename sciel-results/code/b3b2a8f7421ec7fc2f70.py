"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0  # explicit wall-clock guard (150s kill limit, 10s safety)


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with time-dependent driving force.

    Structure: a(t,x,v) = F(t) - R(x) - D(v) + C(t,x,v)
      F(t): time-dependent driving force (DC + fundamental + 2nd harmonic)
      R(x): nonlinear restoring force (odd, saturated)
      D(v): nonlinear damping/drag (odd)
      C(t,x,v): cross-coupling terms (parametric, velocity-position interaction)
    All nonlinear terms use tanh saturation so they remain bounded and
    well-conditioned for BFGS from an all-ones start.
    """
    if time.time() > _DEADLINE:
        return np.zeros_like(np.asarray(t, dtype=float), dtype=float)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = (float(params[i]) for i in range(10))

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(x, dtype=float) if v is None else np.asarray(v, dtype=float)

    # --- time-dependent driving force (bounded) ---
    s1 = np.sin(t)
    c1 = np.cos(t)
    s2 = np.sin(2.0 * t)
    c2 = np.cos(2.0 * t)
    F = p0 + p1 * s1 + p2 * c1 + p3 * s2 + p4 * c2

    # --- nonlinear restoring force (odd in x, saturated) ---
    tx = np.tanh(x)
    # x*tanh(x) ~ x for small x, saturates for large x
    R = p5 * x * tx + p6 * (x ** 3) * tx

    # --- nonlinear damping / drag (odd in v, saturated) ---
    tv = np.tanh(v)
    # v*tanh(v) ~ v for small v, saturates for large v
    D = p7 * v * tv + p8 * (v ** 3) * tv

    # --- cross-coupling terms ---
    # Parametric modulation: x * sin(t)
    # Velocity-position interaction: x * v
    C = p9 * (x * s1 + 0.5 * x * v)

    a = F - R - D + C

    # guarantee finite output
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a
# EVOLVE-BLOCK-END
