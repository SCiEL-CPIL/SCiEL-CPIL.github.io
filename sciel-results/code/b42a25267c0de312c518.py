"""Dispatch rule for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = 120.0
_START = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _START


def _safe_dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with the shortest duration for this op.
        best_m = min(op["machine_options"].items(), key=lambda kv: kv[1])[0]
        return (0, best_m)

    # Composite scoring:
    #   - remaining_work: long jobs should start early (minimize makespan)
    #   - processing time on best machine: shorter is better (reduces idle)
    #   - remaining_ops: jobs with fewer remaining ops are closer to done
    #
    # We want to maximize a score that rewards:
    #   high remaining_work, low min_processing_time, low remaining_ops
    # Normalize each feature to [0,1] and combine with weights.

    rw = [op["remaining_work"] for op in ready_operations]
    ro = [op["remaining_ops"] for op in ready_operations]
    pt = [min(op["machine_options"].values()) for op in ready_operations]

    rw_min, rw_max = min(rw), max(rw)
    ro_min, ro_max = min(ro), max(ro)
    pt_min, pt_max = min(pt), max(pt)

    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0
    pt_range = pt_max - pt_min if pt_max > pt_min else 1.0

    # Normalize to [0,1]
    rw_norm = [(v - rw_min) / rw_range for v in rw]
    ro_norm = [(v - ro_min) / ro_range for v in ro]
    pt_norm = [(v - pt_min) / pt_range for v in pt]

    # Weights: emphasize remaining work most, then processing time, then remaining ops
    w_rw = 0.50
    w_pt = 0.30
    w_ro = 0.20

    # Score: higher is better.
    #   rw_norm: higher remaining work -> higher score (start long jobs early)
    #   pt_norm: lower processing time -> higher score (inverse)
    #   ro_norm: lower remaining ops -> higher score (inverse, finish near-done jobs)
    scores = []
    for i in range(n):
        s = w_rw * rw_norm[i] + w_pt * (1.0 - pt_norm[i]) + w_ro * (1.0 - ro_norm[i])
        scores.append(s)

    best_idx = max(range(n), key=lambda i: scores[i])
    op = ready_operations[best_idx]
    best_m = min(op["machine_options"].items(), key=lambda kv: kv[1])[0]
    return (best_idx, best_m)


def dispatch(ready_operations, time):
    # Deadline guard: if we're getting close to the limit, fall back to a fast rule.
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: pick the operation with the most remaining work.
        best = max(range(len(ready_operations)), key=lambda i: ready_operations[i]["remaining_work"])
        op = ready_operations[best]
        best_m = min(op["machine_options"].items(), key=lambda kv: kv[1])[0]
        return (best, best_m)

    return _safe_dispatch(ready_operations, time)
# EVOLVE-BLOCK-END
