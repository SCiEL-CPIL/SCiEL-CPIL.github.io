"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 145.0
_MACHINE_LOAD = {}


def _elapsed():
    return _time.time() - _dispatch_start_time


def _soft_min(durations, temperature=0.5):
    """
    Smooth surrogate for min(durations) using log-sum-exp.
    Returns a value close to min but differentiable/smooth, useful for
    near-tie machine selection without hard discontinuities.
    """
    if not durations:
        return float('inf')
    arr = np.array(durations, dtype=float)
    mn = np.min(arr)
    # soft-min = -log-sum-exp(-x / T) * T
    scaled = -arr / temperature
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -lse * temperature


def _machine_load(machine):
    """
    Number of ready operations that could run on this machine (congestion proxy).
    """
    return _MACHINE_LOAD.get(machine, 0)


def dispatch(ready_operations, time):
    """
    Dispatch rule: urgency (remaining work) + efficiency (soft-min duration)
    + load balancing (fewer competing ops on chosen machine).
    Returns (index, machine).
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, pick a quick heuristic
    if _elapsed() > _DEADLINE:
        # Fallback: pick operation with max remaining work, min raw duration
        best_i, best_m = 0, None
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            if rw > best_rw:
                best_rw = rw
                best_i = i
                best_m = min(op["machine_options"], key=op["machine_options"].get)
        if best_m is None:
            best_m = next(iter(ready_operations[0]["machine_options"]))
        return (best_i, best_m)

    # Update machine load counts based on current ready set
    _MACHINE_LOAD.clear()
    for op in ready_operations:
        for m in op["machine_options"]:
            _MACHINE_LOAD[m] = _MACHINE_LOAD.get(m, 0) + 1

    best_idx = -1
    best_machine = None
    best_score = -float('inf')

    # Weights: urgency dominates, then efficiency, then load balance
    W_URGENCY = 1.0
    W_EFFICIENCY = 10.0
    W_LOAD = 1.0

    # Normalize urgency by max remaining work for stability
    max_rw = max(op["remaining_work"] for op in ready_operations) if ready_operations else 1
    if max_rw <= 0:
        max_rw = 1

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        durations = list(machine_options.values())
        machines = list(machine_options.keys())

        # Soft-min duration (smooth surrogate for best machine time)
        sm_dur = _soft_min(durations, temperature=0.5)

        # Efficiency: lower duration is better -> use -sm_dur
        efficiency = -sm_dur

        # Load balance: prefer machines with fewer competing ready ops
        # Use the machine that achieves the soft-min as the "chosen" machine
        # Find the actual machine with min duration (tie-break by load)
        min_dur = float('inf')
        chosen_machine = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                chosen_machine = m
            elif d == min_dur and chosen_machine is not None:
                if _machine_load(m) < _machine_load(chosen_machine):
                    chosen_machine = m
        if chosen_machine is None:
            chosen_machine = machines[0]

        load_penalty = float(_machine_load(chosen_machine))

        urgency = op["remaining_work"] / max_rw

        score = (W_URGENCY * urgency
                 + W_EFFICIENCY * efficiency
                 - W_LOAD * load_penalty)

        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_machine

    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
