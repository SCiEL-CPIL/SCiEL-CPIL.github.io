"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 145.0


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _min_machine(opts):
    best_m = None
    best_d = float('inf')
    for m, d in opts.items():
        if d < best_d or (d == best_d and best_m is not None and m < best_m):
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

    # Precompute features
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    for op in ready_operations:
        opts = op["machine_options"]
        bm, bd = _min_machine(opts)
        best_machines.append(bm)
        min_durs.append(bd)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])

    # Deadline guard
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Normalize features to [0, 1]
    rw = np.array(rem_works, dtype=float)
    md = np.array(min_durs, dtype=float)
    ro = np.array(rem_ops, dtype=float)

    rw_norm = (rw - rw.min()) / (rw.max() - rw.min() + 1e-9)
    md_norm = (md - md.min()) / (md.max() - md.min() + 1e-9)
    ro_norm = (ro - ro.min()) / (ro.max() - ro.min() + 1e-9)

    # Weighted score: maximize MWR, minimize SPT, minimize remaining ops
    # Weights: MWR is dominant, then SPT, then remaining ops
    scores = 1.0 * rw_norm - 0.3 * md_norm - 0.2 * ro_norm

    idx = int(np.argmax(scores))
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
