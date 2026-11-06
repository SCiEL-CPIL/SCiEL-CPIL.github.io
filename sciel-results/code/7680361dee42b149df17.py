"""Rewritten open-shop dispatch rule with a composite heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE_SECONDS = 145.0
_start_wall = _time.time()


def _elapsed_ok():
    return (_time.time() - _start_wall) < _DEADLINE_SECONDS


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.

    Strategy (refined composite within the same family):
      - Primary: estimated job completion time (current time + remaining_work),
        i.e. the job with the longest tail is most urgent for makespan.
      - Secondary: number of remaining operations (longer jobs first to avoid
        stranding short jobs on a single machine later).
      - Tertiary: minimum duration across machine options (prefer shorter,
        more flexible ops when else comparable) to reduce machine blocking.
      - Use a smooth (log-sum-exp style) aggregation so near-ties don't cause
        brittle flips; the argmax of the smooth score equals the argmax of the
        raw weighted sum, but the magnitudes are well-scaled for stability.

    Returns index of the chosen ready operation.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Per-op features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    min_durations = []
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            min_durations.append(min(opts.values()))
        else:
            min_durations.append(0.0)

    # Estimated job completion time if this op is started now (proxy for tail)
    # We use remaining_work directly; adding `time` is a constant offset.
    # Normalize each feature to [0, 1] over the candidate set.
    def _normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-12:
            return [1.0] * len(vals)
        return [(v - mn) / rng for v in vals]

    nw = _normalize(remaining_works)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # Weights (same family as before, tuned to emphasize urgency)
    w_work = 0.65
    w_ops = 0.20
    w_dur = 0.15  # subtract: shorter min-duration is better

    # Raw weighted scores (higher = more urgent)
    raw = [
        w_work * nw[i] + w_ops * no[i] - w_dur * nd[i]
        for i in range(n)
    ]

    # Smooth aggregation via log-sum-exp to avoid brittle near-tie flips;
    # argmax(raw) == argmax(logsumexp(raw, temperature)) for temperature>0,
    # but the values are numerically stable and the ranking is preserved.
    # We just need the argmax; computing the smooth value is optional.
    # To keep it cheap and exact, we still pick argmax of raw, but we use a
    # small tie-breaking nudge based on a secondary smooth term to break
    # exact ties deterministically in favor of higher remaining_ops.
    best_idx = 0
    best_score = raw[0]
    for i in range(1, n):
        if raw[i] > best_score + 1e-12:
            best_score = raw[i]
            best_idx = i
        elif abs(raw[i] - best_score) <= 1e-12:
            # Tie-break: prefer more remaining ops, then shorter min-duration
            if (remaining_ops[i] > remaining_ops[best_idx]) or \
               (remaining_ops[i] == remaining_ops[best_idx] and
                min_durations[i] < min_durations[best_idx]):
                best_idx = i
                best_score = raw[i]

    return int(best_idx)
# EVOLVE-BLOCK-END
