"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
from math import log, exp

_DEADLINE = 110.0  # Safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.

    Strategy:
    1. Rank ready operations by a priority that combines:
       - Most-Work-Remaining (MRW): jobs with more remaining work are critical.
       - Shortest-Processing-Time (SPT): faster ops finish sooner and free machines.
       - Completion proximity: jobs with fewer remaining ops are near done, so
         finishing them releases downstream operations.
    2. For the chosen operation, pick the machine that yields the earliest
       finish time (simulated from the current time plus the op's duration),
       breaking ties by shorter duration.

    Returns (index, machine) or index.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    if _elapsed() > _DEADLINE:
        # Hard safety fallback: SPT on best available machine
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op.get("machine_options", {})
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    # Tuning parameters
    MRW_WEIGHT = 1.0
    SPT_WEIGHT = 0.5
    COMPLETION_WEIGHT = 0.25

    best_score = -float('inf')
    best_index = 0
    best_min_dur = float('inf')

    for i, op in enumerate(ready_operations):
        opts = op.get("machine_options", {})
        if not opts:
            continue
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1.0))
        if rw <= 0:
            continue

        min_dur = min(opts.values())
        if min_dur <= 0:
            min_dur = 1e-6

        # MRW component: scale remaining work by its own magnitude (favors big jobs)
        mrw = rw * rw
        # SPT component: favor shorter processing
        spt = 1.0 / min_dur
        # Completion proximity: fewer remaining ops -> higher urgency
        comp = 1.0 / max(ro, 1.0)

        score = MRW_WEIGHT * mrw + SPT_WEIGHT * spt + COMPLETION_WEIGHT * comp

        if score > best_score or (score == best_score and min_dur < best_min_dur):
            best_score = score
            best_index = i
            best_min_dur = min_dur

    # Pick the machine for the chosen operation: earliest finish from current time.
    chosen = ready_operations[best_index]
    opts = chosen.get("machine_options", {})
    if opts:
        # All machines are available at `time` (the simulation hands us ready ops at
        # the current event time), so earliest finish == shortest duration.
        best_machine = min(opts, key=lambda m: (opts[m], m))
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
