"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from math import log, exp, isinf

_START_TIME = _time.time()
_DEADLINE = 120.0
_MARGIN = 12.0  # seconds of safety margin


def _elapsed() -> float:
    return _time.time() - _START_TIME


def _soft_max(values, temp=1.0):
    """Soft-max via log-sum-exp for smooth surrogate of max."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    m = np.max(arr)
    if isinf(m):
        return float(m)
    if m > 500:
        # Avoid overflow in exp
        shifted = arr - m
        log_sum = np.log(np.sum(np.exp(shifted / temp)))
        return float(m + temp * log_sum)
    return float(np.log(np.sum(np.exp(arr / temp))) * temp)


def _soft_min(values, temp=1.0):
    """Soft-min via -soft_max(-values)."""
    if not values:
        return 0.0
    return -_soft_max([-v for v in values], temp)


def _best_machine_for_op(op, time, ready_operations):
    """
    Select the best machine for a given operation.
    
    Strategy:
    - Primary: minimize the operation's own duration on that machine (frees it sooner).
    - Secondary: among ties, prefer the machine that is "least loaded" in the sense
      of having fewer other ready operations that could also use it (reduces conflict).
    
    Returns (machine, duration).
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    
    # Count how many other ready operations also list this machine
    machine_conflict = {}
    for m, d in durations.items():
        conflict = 0
        for other_op in ready_operations:
            if other_op is op:
                continue
            if m in other_op.get("machine_options", {}):
                conflict += 1
        machine_conflict[m] = conflict
    
    # Pick machine with min duration; tie-break by min conflict
    best_m = min(durations.keys(), key=lambda m: (durations[m], machine_conflict.get(m, 0)))
    return best_m, durations[best_m]


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy (refined within the same family):
    1. Primary signal: job with the most remaining work (criticality).
       We use a soft-max over remaining_work to get a smooth surrogate,
       but since we're picking a single op, we just rank by remaining_work.
    
    2. Secondary signal: among operations, prefer those where the chosen
       machine's duration is longest (MPT) — this commits the machine to a
       long block early, reducing fragmentation and tail effects.
       We use soft-max as a smooth surrogate for the max duration.
    
    3. Tertiary: prefer operations with fewer remaining_ops (closer to completion),
       as finishing a job frees its entire sequence of future operations.
    
    4. Machine selection: for the chosen operation, pick the machine that
       minimizes duration (SPT), with tie-breaking by lower conflict count.
    
    Composite score (higher is better):
        score = remaining_work * W1 + soft_max(durations, temp) * W2 - remaining_ops * W3
    
    We return (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op, time, ready_operations)
        return (0, best_m)

    # Deadline guard
    if _elapsed() > _DEADLINE - _MARGIN:
        # Fallback: just pick the one with most remaining work
        rw_list = [op.get("remaining_work", 0) for op in ready_operations]
        best_idx = int(np.argmax(rw_list))
        best_m, _ = _best_machine_for_op(ready_operations[best_idx], time, ready_operations)
        return (best_idx, best_m)

    # Compute scores
    scores = []
    best_machines = []

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        durations = op.get("machine_options", {})
        durs = list(durations.values()) if durations else [0]
        
        # Soft-max of durations as smooth surrogate for max duration (MPT preference)
        smax = _soft_max(durs, temp=2.0)
        
        # Remaining ops count (fewer is better — closer to done)
        rem_ops = op.get("remaining_ops", 0)
        
        # Composite score
        # remaining_work dominates (criticality)
        # soft_max duration as secondary (MPT)
        # remaining_ops as tertiary (prefer fewer remaining ops)
        score = rw * 1e6 + smax * 1e3 - rem_ops * 1e2
        
        scores.append(score)
        best_m, best_d = _best_machine_for_op(op, time, ready_operations)
        best_machines.append(best_m)

    scores_arr = np.asarray(scores, dtype=np.float64)
    best_idx = int(np.argmax(scores_arr))
    best_machine = best_machines[best_idx]
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
