import time as _time
import numpy as np

_START = _time.monotonic()
_DEADLINE = 120.0  # explicit safety margin below the 150 s wall-clock kill

def _deadline_exceeded():
    return (_time.monotonic() - _START) > _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Explicit deadline guard: fall back to the cheapest valid heuristic.
    if _deadline_exceeded():
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    rw = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    ro = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=float)

    dmin = np.full(n, np.inf)
    dmax = np.full(n, 0.0)
    flex = np.full(n, 0.0)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            dmin[i] = 0.0
            continue
        durs = list(mo.values())
        dmin[i] = min(durs)
        dmax[i] = max(durs)
        flex[i] = len(durs)

    dmin_safe = np.where(dmin > 0, dmin, 1.0)
    dmax_safe = np.where(dmax > 0, dmax, 1.0)

    speed = 1.0 / dmin_safe          # finish fast -> frees machine sooner
    shortness = 1.0 / dmax_safe      # bounded worst-case duration
    rigidity = 1.0 / np.where(flex > 0, flex, 1.0)  # few options -> schedule now

    def norm(x):
        x = np.asarray(x, dtype=float)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    score = (0.40 * norm(rw)
             + 0.20 * norm(ro)
             + 0.15 * norm(speed)
             + 0.15 * norm(rigidity)
             + 0.10 * norm(shortness))

    return int(np.argmax(score))
