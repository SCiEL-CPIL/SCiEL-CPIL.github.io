"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Primary: Job with most remaining work (LRPT) to prioritize critical path.
    2. Secondary: Operation with shortest minimum duration (SPT) to release resources quickly.
    3. Machine: Machine with shortest duration for the chosen operation (SPT).
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    # Deadline guard: if we are too close to the limit, return a quick heuristic
    if _elapsed() > 145:
        # Quick fallback: pick first op, min duration machine
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    scores = []
    best_machines = []
    
    # Aggregate machine load across ready ops to prefer less-loaded machines on ties.
    # For each machine, sum of min durations across ready ops that can use it.
    machine_load = {}
    for op in ready_operations:
        durations = op["machine_options"]
        if not durations:
            continue
        for m, d in durations.items():
            machine_load[m] = machine_load.get(m, 0.0) + d

    for op in ready_operations:
        rw = op["remaining_work"]
        durations = op["machine_options"]
        if not durations:
            scores.append(-1)
            best_machines.append(None)
            continue

        # Smooth surrogate for min duration via log-sum-exp (soft-min).
        # Lower soft-min => higher score. Use a moderate temperature so it's
        # close to true min but differentiable/smooth.
        vals = np.array(list(durations.values()), dtype=float)
        # soft-min = -logsumexp(-vals / T) * T
        T = 1.0
        neg = -vals / T
        neg_shift = neg - np.max(neg)
        logsumexp_neg = np.max(neg) + np.log(np.sum(np.exp(neg_shift)))
        soft_min_dur = -T * logsumexp_neg

        # Primary: remaining work (LRPT). Secondary: soft-min duration (SPT-like).
        # Tertiary: prefer machine with lower aggregate load (spread load).
        best_m, best_d = _best_machine_for_op(op)
        load_penalty = machine_load.get(best_m, 0.0) * 1e-3

        score = rw * 1e6 - soft_min_dur - load_penalty
        scores.append(score)
        best_machines.append(best_m)

    # Find index with max score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is None:
        return best_idx
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END
