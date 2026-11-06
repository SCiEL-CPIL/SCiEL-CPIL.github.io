"""Rewritten open-shop dispatch rule with a composite heuristic."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE_SECONDS = 145.0
_start_wall = _time.time()


def _elapsed_ok():
    return (_time.time() - _start_wall) < _DEADLINE_SECONDS


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.

    Strategy:
      - Prefer operations with the most remaining work (longest job tail).
      - Among ties / near-ties, prefer operations that have more remaining
        operations (longer job) to avoid leaving short jobs stranded.
      - Use the minimum processing time across machine options as a tie-breaker
        in favor of shorter, more flexible operations when the above are close.
      - Combine into a weighted score; higher score wins.

    The weights are chosen to emphasize remaining_work (dominant for makespan)
    while using remaining_ops and min_duration as secondary signals.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Compute per-op features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    min_durations = []
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            min_durations.append(min(opts.values()))
        else:
            min_durations.append(0)

    # Normalize features to [0, 1] for stable weighting
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

    # Weights: remaining work dominates, remaining ops secondary,
    # min duration used as a mild tie-breaker (prefer shorter when else equal)
    w_work = 0.70
    w_ops = 0.20
    w_dur = 0.10  # lower duration is better, so subtract

    scores = [
        w_work * nw[i] + w_ops * no[i] - w_dur * nd[i]
        for i in range(n)
    ]

    # Return index of max score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
