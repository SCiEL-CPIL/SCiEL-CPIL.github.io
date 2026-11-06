"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _min_machine(opts):
    """Return (machine, duration) for the machine with minimum duration, tie-break by machine id."""
    best_m = None
    best_d = float('inf')
    for m, d in opts.items():
        if d < best_d or (d == best_d and best_m is not None and m < best_m):
            best_d = d
            best_m = m
    return best_m, best_d


def _max_machine(opts):
    """Return (machine, duration) for the machine with maximum duration, tie-break by machine id."""
    best_m = None
    best_d = -float('inf')
    for m, d in opts.items():
        if d > best_d or (d == best_d and best_m is not None and m > best_m):
            best_d = d
            best_m = m
    return best_m, best_d


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m, _ = _min_machine(opts)
        return (0, best_m)

    # Precompute for each operation:
    # - best machine (min duration)
    # - worst machine (max duration)
    # - spread (max - min duration)
    # - remaining work, remaining ops
    best_machines = []
    min_durs = []
    max_durs = []
    spreads = []
    rem_works = []
    rem_ops = []
    for op in ready_operations:
        opts = op["machine_options"]
        bm, bd = _min_machine(opts)
        wm, wd = _max_machine(opts)
        best_machines.append(bm)
        min_durs.append(bd)
        max_durs.append(wd)
        spreads.append(wd - bd)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])

    # Deadline guard
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Composite scoring for each operation:
    # Primary: maximize remaining work (MWR) to reduce makespan by prioritizing long jobs
    # Secondary: minimize spread (choose ops where machine choice matters less, or
    #            where the best machine is clearly better) to reduce assignment risk
    # Tertiary: minimize min duration (SPT) to keep machines busy with short tasks
    # Quaternary: minimize remaining ops (finish jobs sooner)
    scores = []
    for i in range(n):
        # Normalize to 0-1 range across all ops for balanced weighting
        # We'll sort by a composite key instead
        scores.append(i)

    # Sort by composite key:
    # 1. -remaining_work (most work first)
    # 2. spread (lower spread = less sensitivity to machine choice, safer)
    # 3. min_duration (shorter is better for SPT)
    # 4. remaining_ops (fewer = closer to completion)
    order = sorted(
        scores,
        key=lambda i: (
            -rem_works[i],
            spreads[i],
            min_durs[i],
            rem_ops[i],
        ),
    )

    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
