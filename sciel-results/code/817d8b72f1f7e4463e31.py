"""Job-Shop dispatch: pick the ready operation to start next to minimize makespan."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Wall-clock deadline guard: evaluator kills after 150s. Stay well under.
_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin

def _machine_load_map(ready_operations):
    """Approximate current machine load: sum of min durations assigned to each machine
    across all ready operations. This is a cheap proxy for which machines are busier."""
    load = {}
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if not opts:
            continue
        # Attribute the operation's shortest duration to its least-loaded (preferred) machine
        # but for a load estimate, count each op's min duration once against its best machine.
        best_m = min(opts, key=lambda m: (opts[m], m))
        load[best_m] = load.get(best_m, 0.0) + opts[best_m]
    return load

def dispatch(ready_operations, time):
    # Deadline guard: if we're near the kill limit, just return a quick choice.
    if _time.time() - _START_TS > DEADLINE_S:
        return 0

    if not ready_operations:
        return 0
    n = len(ready_operations)
    if n == 1:
        return 0

    # Approximate machine loads for load-balancing tie-break.
    mload = _machine_load_map(ready_operations)

    # Precompute per-operation features.
    scores = np.empty(n, dtype=np.float64)

    # Gather raw features to normalize robustly.
    rw = np.empty(n)      # remaining_work
    ro = np.empty(n)      # remaining_ops
    min_dur = np.empty(n) # min duration over machine options
    chosen_dur = np.empty(n)  # duration on the chosen (preferred) machine

    for i, op in enumerate(ready_operations):
        rw[i] = float(op.get("remaining_work", 0.0))
        ro[i] = max(1, int(op.get("remaining_ops", 0)))
        opts = op.get("machine_options", {})
        if not opts:
            min_dur[i] = 1.0
            chosen_dur[i] = 1.0
            continue
        # Preferred machine: shortest duration; tie-break by lower current load, then id.
        best_m = min(opts, key=lambda m: (opts[m], mload.get(m, 0.0), m))
        chosen_dur[i] = opts[best_m]
        min_dur[i] = min(opts.values())
        
        # If there are multiple machines with the same min duration,
        # prefer the one with lowest load to balance the schedule.
        min_d = min_dur[i]
        tied = [m for m, d in opts.items() if d == min_d]
        if len(tied) > 1:
            best_m = min(tied, key=lambda m: (mload.get(m, 0.0), m))
            chosen_dur[i] = opts[best_m]

    # Robust normalization via ranks to avoid scale issues across instances.
    def rank(arr):
        # average rank (1-based) for ties
        order = np.argsort(arr)
        ranks = np.empty_like(arr, dtype=np.float64)
        # assign ranks with tie-averaging
        sorted_vals = arr[order]
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    # We want to FAVOR:
    #  - high remaining_work  -> higher rank is better
    #  - short min_duration   -> lower is better, so use (max_rank - rank + 1)
    #  - few remaining_ops    -> lower is better
    #  - short chosen_dur     -> lower is better

    rw_rank = rank(rw)          # high = good
    dur_rank = rank(min_dur)    # high = long (bad), so invert
    ro_rank = rank(ro)          # high = many (bad), so invert
    cdur_rank = rank(chosen_dur)

    inv_dur = (len(rw) + 1.0) - dur_rank
    inv_ro = (len(rw) + 1.0) - ro_rank
    inv_cdur = (len(rw) + 1.0) - cdur_rank

    # Composite score. Weights tuned to prioritize keeping critical (high-work) jobs
    # moving and avoiding long operations; small weight on load-balanced duration.
    # SPT should be the primary driver for makespan minimization.
    # MRW is a secondary tie-breaker to keep critical jobs moving.
    w_rw = 0.5
    w_dur = 1.5
    w_ro = 0.5
    w_cdur = 0.25

    scores = (w_rw * rw_rank
              + w_dur * inv_dur
              + w_ro * inv_ro
              + w_cdur * inv_cdur)

    best_idx = int(np.argmax(scores))

    # Return just the index (the documented primary interface).
    return best_idx
# EVOLVE-BLOCK-END
