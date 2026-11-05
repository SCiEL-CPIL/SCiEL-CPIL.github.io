"""Job-Shop Scheduling: critical-path + machine-utilization dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 140.0
_START_TIME = _time.monotonic()

# Weights for the composite score (higher = pick).
# Tuned to balance critical-path urgency vs. machine-utilization efficiency.
_W_REM_WORK = 1.0     # critical-path proxy: more remaining work => more urgent
_W_MIN_DUR   = 0.8    # short min-duration => frees machine fast (SPT-like)
_W_REM_OPS   = 0.3    # few remaining ops => job finishes sooner, releases deps
_W_SPREAD    = 0.2    # prefer ops with low min/mean ratio (flexible placement)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time

    Returns:
        index of the ready operation to start
    """
    # Deadline guard: if we're close to the limit, just return first op
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Gather features into arrays
    rem_work = np.zeros(n)
    min_dur  = np.zeros(n)
    mean_dur = np.zeros(n)
    rem_ops  = np.zeros(n)

    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            rem_work[i] = -1.0  # mark as invalid
            continue
        vals = list(machine_opts.values())
        rem_work[i] = op.get("remaining_work", 0.0)
        min_dur[i]  = min(vals)
        mean_dur[i] = float(np.mean(vals))
        rem_ops[i]  = op.get("remaining_ops", 0)

    # Normalize features to [0,1] range for stable weighting
    def _norm(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # For min_dur, we want LOWER to be better, so invert
    norm_rem_work = _norm(rem_work)       # higher = more urgent
    norm_min_dur  = 1.0 - _norm(min_dur)  # higher = shorter (better)
    norm_rem_ops  = 1.0 - _norm(rem_ops)  # higher = fewer ops left (better)
    # spread: min/mean ratio; lower ratio means more flexible (some machines faster)
    with np.errstate(divide='ignore', invalid='ignore'):
        spread = min_dur / np.maximum(mean_dur, 1e-9)
    norm_spread = 1.0 - _norm(spread)     # higher = more flexible

    # Composite score
    scores = (_W_REM_WORK * norm_rem_work
              + _W_MIN_DUR  * norm_min_dur
              + _W_REM_OPS  * norm_rem_ops
              + _W_SPREAD   * norm_spread)

    # Mark invalid ops
    scores[rem_work < 0] = -np.inf

    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
