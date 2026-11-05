"""Job-Shop Scheduling: critical-path + least-loaded-machine dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 140.0
_START_TIME = _time.monotonic()

# Global machine-load tracker: maps machine -> estimated busy time
_machine_load = {}

# Weights
_W_CP   = 1.0   # critical path (remaining work) — primary urgency
_W_LOAD = 0.6   # machine load penalty — congestion avoidance
_W_SPT  = 0.15  # shortest processing time tie-break


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
    # Deadline guard
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # For each ready op, compute:
    #   rem_work[i]   = remaining_work
    #   best_machine[i] = machine with min duration
    #   best_dur[i]   = min duration
    rem_work = np.zeros(n)
    best_dur = np.full(n, np.inf)
    best_mach = [None] * n

    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            rem_work[i] = -1.0
            continue
        rem_work[i] = op.get("remaining_work", 0.0)
        # Find best (shortest) machine
        best_m = None
        best_d = np.inf
        for m, d in machine_opts.items():
            if d < best_d:
                best_d = d
                best_m = m
        best_dur[i] = best_d
        best_mach[i] = best_m

    # Estimate current load for each op's best machine
    # Use max(time, machine_load[m]) as the earliest the machine can start
    load_penalty = np.zeros(n)
    for i in range(n):
        m = best_mach[i]
        if m is not None:
            cur_load = _machine_load.get(m, 0.0)
            # The machine will be busy until max(time, cur_load); 
            # penalty is how much this machine is already loaded beyond now
            load_penalty[i] = max(0.0, cur_load - time)
        else:
            load_penalty[i] = np.inf

    # Normalize features
    def _norm(x):
        valid = x[x >= 0] if (x < 0).any() else x
        if len(valid) == 0:
            return np.zeros_like(x)
        lo, hi = float(np.min(valid)), float(np.max(valid))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # Only normalize over valid entries
    valid_mask = rem_work >= 0
    if not valid_mask.any():
        return 0

    # Normalize rem_work (higher = more urgent = better)
    norm_rw = np.zeros(n)
    rw_valid = rem_work[valid_mask]
    lo, hi = float(np.min(rw_valid)), float(np.max(rw_valid))
    if hi - lo > 1e-12:
        norm_rw[valid_mask] = (rem_work[valid_mask] - lo) / (hi - lo)
    else:
        norm_rw[valid_mask] = 1.0

    # Normalize load_penalty (lower = better, so invert)
    norm_load = np.zeros(n)
    lp_valid = load_penalty[valid_mask]
    lo_l, hi_l = float(np.min(lp_valid)), float(np.max(lp_valid))
    if hi_l - lo_l > 1e-12:
        norm_load[valid_mask] = 1.0 - (load_penalty[valid_mask] - lo_l) / (hi_l - lo_l)
    else:
        norm_load[valid_mask] = 1.0

    # Normalize best_dur (lower = better, invert)
    norm_spt = np.zeros(n)
    bd_valid = best_dur[valid_mask]
    lo_b, hi_b = float(np.min(bd_valid)), float(np.max(bd_valid))
    if hi_b - lo_b > 1e-12:
        norm_spt[valid_mask] = 1.0 - (best_dur[valid_mask] - lo_b) / (hi_b - lo_b)
    else:
        norm_spt[valid_mask] = 1.0

    # Composite score
    scores = (_W_CP * norm_rw
              + _W_LOAD * norm_load
              + _W_SPT * norm_spt)

    # Invalid ops get -inf
    scores[~valid_mask] = -np.inf

    idx = int(np.argmax(scores))

    # Update machine load for the chosen op
    m = best_mach[idx]
    if m is not None:
        cur = _machine_load.get(m, 0.0)
        # Machine will be busy from max(time, cur) for best_dur[idx]
        start = max(time, cur)
        _machine_load[m] = start + best_dur[idx]

    return idx
# EVOLVE-BLOCK-END
