"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time as _time

# ---- Deadline guard ----
_dispatch_start = _time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

# ---- Tunable hyperparameters ----
W_REM_WORK = 1.0      # weight on remaining_work (urgency)
W_TAIL     = 0.55     # weight on remaining_ops (critical-path tail urgency)
W_EFF      = 0.65     # weight on efficiency (fastest machine)
W_LOAD     = 0.45     # weight on least-loaded machine (contention)
W_BOTTLE   = 0.5      # weight on bottleneck look-ahead
LSE_BETA   = 10.0     # sharpness of soft-min

# ---- Machine load tracking (global state) ----
_machine_load = {}
_machine_count = {}


def _soft_min(durations):
    """Smooth log-sum-exp surrogate for min over positive durations."""
    if not durations:
        return 0.0
    if len(durations) == 1:
        return durations[0]
    neg = [-LSE_BETA * d for d in durations]
    m = max(neg)
    s = math.exp(m) + sum(math.exp(x - m) for x in neg[1:])
    return -(m + math.log(s)) / LSE_BETA


def _best_duration(op):
    """Fastest (minimum) duration among eligible machines."""
    opts = op.get("machine_options", {})
    if not opts:
        return float("inf")
    return min(opts.values())


def _least_loaded_machine(op):
    """Pick machine with smallest (duration + current_load) for this op."""
    opts = op.get("machine_options", {})
    if not opts:
        return None
    best_machine = None
    best_key = float("inf")
    for machine, dur in opts.items():
        load = _machine_load.get(machine, 0.0)
        key = dur + load
        if key < best_key:
            best_key = key
            best_machine = machine
    return best_machine


def _earliest_completion(op, current_time):
    """Estimate earliest completion time considering current machine loads."""
    opts = op.get("machine_options", {})
    if not opts:
        return float("inf")
    best = float("inf")
    for machine, dur in opts.items():
        load = _machine_load.get(machine, 0.0)
        comp = current_time + load + dur
        if comp < best:
            best = comp
    return best


def _norm(vals):
    """Normalize a list of values to [0, 1]."""
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span <= 1e-12:
        return [0.5] * len(vals)
    return [(v - lo) / span for v in vals]


def dispatch(ready_operations, time):
    global _machine_load, _machine_count

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        m = _least_loaded_machine(op)
        if m is not None:
            _machine_load[m] = _machine_load.get(m, 0.0) + op.get("machine_options", {}).get(m, 0.0)
            return (0, m)
        return 0

    # Deadline check
    if _time.time() - _dispatch_start > _DEADLINE:
        return 0

    # ---- Per-operation signals ----
    rem_works = []
    tails = []
    best_durs = []
    soft_durs = []
    completions = []

    for op in ready_operations:
        rem_works.append(float(op.get("remaining_work", 0.0)))
        tails.append(float(op.get("remaining_ops", 0.0)))
        bd = _best_duration(op)
        best_durs.append(bd)
        soft_durs.append(_soft_min(list(op.get("machine_options", {}).values())))
        completions.append(_earliest_completion(op, time))

    # ---- Normalize ----
    nw = _norm(rem_works)
    nt = _norm(tails)
    eff_raw = [(-d if d < float("inf") else 0.0) for d in best_durs]
    ne = _norm(eff_raw)
    load_raw = [(-d if d < float("inf") else 0.0) for d in soft_durs]
    nl = _norm(load_raw)
    comp_raw = [(-c if c < float("inf") else 0.0) for c in completions]
    nb = _norm(comp_raw)

    # ---- Combined score ----
    scores = [
        W_REM_WORK * nw[i]
        + W_TAIL * nt[i]
        + W_EFF * ne[i]
        + W_LOAD * nl[i]
        + W_BOTTLE * nb[i]
        for i in range(n)
    ]

    # ---- Pick best index ----
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-12:
            best_score = scores[i]
            best_idx = i

    # ---- Choose machine and update load ----
    chosen = ready_operations[best_idx]
    m = _least_loaded_machine(chosen)
    if m is not None:
        dur = chosen.get("machine_options", {}).get(m, 0.0)
        _machine_load[m] = _machine_load.get(m, 0.0) + dur
        _machine_count[m] = _machine_count.get(m, 0) + 1
        return (best_idx, m)
    return best_idx
# EVOLVE-BLOCK-END
