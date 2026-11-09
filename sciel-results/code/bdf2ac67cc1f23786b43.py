"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = 145.0  # safety margin before 150s kill
_start_wall = _time.time()


def _elapsed_ok():
    return (_time.time() - _start_wall) < _DEADLINE


def dispatch(ready_operations, time):
    """Pick which ready operation to start next and which machine to run it on.

    Returns (index, machine) where machine is the key from machine_options.
    Strategy:
      1. For each ready op, compute best machine = argmin duration.
      2. Score each op by a combination of:
         - urgency: 1 / (remaining_work)  (smaller remaining work -> higher urgency)
         - work load: remaining_work (bigger jobs first to avoid tail effects)
         - machine load balance: prefer machines with smaller min duration
         - operation index (tie-breaker)
      3. Use a weighted composite score.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        best_mach = min(op["machine_options"].items(), key=lambda kv: kv[1])[0]
        return (0, best_mach)

    # Precompute per-operation features
    best_durs = []
    remaining_works = []
    remaining_ops_counts = []
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        best_dur = min(mo.values())
        best_durs.append(best_dur)
        remaining_works.append(op.get("remaining_work", 0.0))
        remaining_ops_counts.append(op.get("remaining_ops", 0))

    # Normalize features to [0, 1] range for stable weighting
    def _normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-12:
            return [0.5] * len(vals)
        return [(v - mn) / rng for v in vals]

    norm_work = _normalize(remaining_works)
    norm_dur = _normalize(best_durs)  # smaller duration -> lower norm
    norm_rem_ops = _normalize(remaining_ops_counts)

    # Composite score: we want to pick the op that balances:
    #   - high remaining work (to avoid long tail)
    #   - short best duration (to keep machines busy quickly)
    #   - few remaining ops (urgency: finish jobs that are close to done)
    scores = []
    for i in range(n):
        s = (
            0.45 * norm_work[i]       # prioritize big jobs
          + 0.30 * (1.0 - norm_dur[i]) # prefer short durations
          + 0.25 * (1.0 - norm_rem_ops[i]) # prefer jobs close to completion
        )
        scores.append(s)

    # Pick the max score
    best_idx = max(range(n), key=lambda i: scores[i])

    # Choose the machine with the shortest duration for that op
    op = ready_operations[best_idx]
    mo = op["machine_options"]
    best_mach = min(mo.items(), key=lambda kv: kv[1])[0]

    return (best_idx, best_mach)
# EVOLVE-BLOCK-END
