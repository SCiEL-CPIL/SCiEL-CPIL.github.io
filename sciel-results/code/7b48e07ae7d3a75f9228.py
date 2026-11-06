"""Rewritten open-shop dispatch rule with a composite urgency/duration score."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Safety margin before the 150s wall-clock kill; track per-call start.
_DEADLINE_SECONDS = 145.0
_call_start = None


def dispatch(ready_operations, time):
    global _call_start
    if _call_start is None:
        _call_start = _time.monotonic()
    if _time.monotonic() - _call_start > _DEADLINE_SECONDS:
        # Safety margin: return the first ready operation.
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Build per-operation features and count machine contention.
    remaining_work = np.empty(n, dtype=float)
    min_dur = np.empty(n, dtype=float)
    avg_dur = np.empty(n, dtype=float)
    scarcity = np.empty(n, dtype=float)
    machine_counts = {}
    for i, op in enumerate(ready_operations):
        remaining_work[i] = float(op["remaining_work"])
        opts = op["machine_options"]
        vals = list(opts.values()) if opts else [0.0]
        min_dur[i] = float(min(vals))
        avg_dur[i] = float(sum(vals) / len(vals))
        for m in opts:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if opts:
            scarcity[i] = float(np.mean([machine_counts[m] for m in opts]))
        else:
            scarcity[i] = 1.0

    # Composite score: balance job criticality (remaining work), short
    # processing time (SPT via min/avg duration), and machine scarcity
    # (prefer operations on less-contended machines to reduce blocking).
    # Log-sum-exp style smooth aggregation keeps the objective differentiable
    # in spirit and scale-robust.
    eps = 1e-9
    # Urgency density: high work / short time
    urgency = remaining_work / (min_dur + eps)
    # SPT component: shorter average duration is better
    spt = 1.0 / (avg_dur + eps)
    # Scarcity: fewer ready ops on same machines is better
    scar = 1.0 / (scarcity + eps)

    # Weighted geometric mean (log-space sum) for smooth multi-criteria
    w1, w2, w3 = 0.5, 0.3, 0.2
    log_score = (w1 * np.log(urgency + eps) +
                 w2 * np.log(spt + eps) +
                 w3 * np.log(scar + eps))

    # Deadline guard: if computation is taking too long, fall back to a
    # simple rule rather than risk a timeout.
    if _time.monotonic() - _call_start > _DEADLINE_SECONDS:
        return 0

    idx = int(np.argmax(log_score))
    return idx
# EVOLVE-BLOCK-END
