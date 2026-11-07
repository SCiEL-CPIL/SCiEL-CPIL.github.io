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
    """Nonlinear Duffing oscillator with multi-harmonic sinusoidal driving force.

    The 10 parameters are physically meaningful constants:
      p0, p1, p2 : linear, quadratic, cubic stiffness coefficients
      p3, p4     : linear, quadratic (velocity-squared) damping coefficients
      p5, p6     : fundamental driving amplitude and phase
      p7, p8     : second-harmonic driving amplitude and phase
      p9         : bias / constant offset

    The model is nonlinear in the parameters (products of stiffness and damping
    coefficients with state variables, parameterized trigonometric arguments),
    which lets BFGS fit a much richer function class than a fixed linear basis.
    """
    # Safety margin guard (closed-form eval is fast; this is a no-op safeguard).
    if not _elapsed_ok():
        # Return a finite fallback of the right shape to avoid non-finite penalties.
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    # Defensive: if params is short, pad with zeros; if long, ignore extras.
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9) = p

    # Nonlinear stiffness (Duffing): linear + quadratic + cubic terms
    # Quadratic term can model asymmetric restoring force
    stiffness = p0 * x + p1 * x * x + p2 * x * x * x

    # Damping: linear (viscous) + quadratic (turbulent) drag
    damping = p3 * v + p4 * v * v

    # Multi-harmonic sinusoidal driving force
    # Fundamental frequency 1.0, second harmonic 2.0
    drive = p5 * np.sin(t + p6) + p7 * np.sin(2.0 * t + p8)

    # Constant bias / offset
    bias = p9

    # Total acceleration
    out = stiffness + damping + drive + bias

    # Guarantee finiteness and correct shape (one prediction per row)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
# EVOLVE-BLOCK-END
