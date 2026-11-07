# EVOLVE-BLOCK-START
import time
import numpy as np

# Fixed, well-conditioned localized Gaussian basis (10 centers over [-3, 3]).
_N = 10
_CENTERS = np.linspace(-3.0, 3.0, _N)          # evenly spaced centers
_WIDTH = 6.0 / (_N - 1)                         # spacing 0.666... -> ~2 sigma overlap


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    t0 = time.time()
    DEADLINE = t0 + 115.0          # explicit wall-clock safety margin (< 120 s)

    x = np.asarray(x, dtype=np.float64)
    x_orig_shape = x.shape
    x_flat = x.ravel()

    # Build the (N, m) Gaussian basis matrix once per call.
    d = x_flat[None, :] - _CENTERS[:, None]
    B = np.exp(-0.5 * (d / _WIDTH) ** 2)        # (N, m)

    # Forward map: linear combination of the localized basis functions.
    p = np.asarray(params, dtype=np.float64).ravel()
    if p.shape[0] != _N:
        p = np.resize(p, _N)

    out = (B @ p).reshape(x_orig_shape)

    # Guarantee finite, correctly-shaped output (invalid shape/non-finite -> bad).
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)

    # Deadline guard: forward pass is O(N*m), trivially fast; safety no-op.
    if time.time() > DEADLINE:
        return out

    return out
# EVOLVE-BLOCK-END
