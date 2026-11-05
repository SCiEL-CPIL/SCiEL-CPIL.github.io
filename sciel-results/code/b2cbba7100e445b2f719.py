"""Open-Shop Scheduling dispatch rule using a contention-aware scoring approach."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Returns (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Max remaining work, Min duration
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        op = ready_operations[best_idx]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (best_idx, best_m)

    if n == 1:
        op = ready_operations[0]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (0, best_m)

    # 1. Calculate contention for each machine
    # Contention is the number of ready operations that can use this machine.
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    # Parameters
    # Lambda: weight of the "effective duration" penalty.
    # Beta: weight of the contention factor.
    LAMBDA = 0.8
    BETA = 0.5
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        durations = op["machine_options"]
        
        if not durations:
            continue

        # Find the best machine for this operation based on effective cost
        # Cost = duration * (1 + beta * contention)
        # We want to minimize this cost.
        min_cost = float('inf')
        best_m_for_op = 0
        
        for m, d in durations.items():
            cont = machine_contention.get(m, 0)
            # Subtract 1? No, contention includes this op. 
            # If this op is the only one using m, contention is 1.
            # If many use it, contention is high.
            cost = d * (1.0 + BETA * cont)
            if cost < min_cost:
                min_cost = cost
                best_m_for_op = m

        # Calculate score for this operation
        # Score = remaining_work - LAMBDA * min_cost
        # This balances criticality (remaining work) against the "cost" of scheduling it
        # (which is higher if it takes long on a busy machine).
        score = rw - LAMBDA * min_cost
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
