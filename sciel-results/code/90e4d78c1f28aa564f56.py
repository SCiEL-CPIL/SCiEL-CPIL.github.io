"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)
_START = _time.monotonic()

_LAMBDA = 0.3  # regret weight

def dispatch(ready_operations, time):
    """
    Regret-based dispatch: pick the ready operation whose assignment would
    cause the least additional delay to other ready operations (regret),
    weighted against remaining_work to prioritize long jobs.
    
    Returns (index, machine) to start.
    """
    # Deadline guard
    if _time.monotonic() - _START > _DEADLINE:
        if not ready_operations:
            return 0
        op = ready_operations[0]
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (0, best_m)
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (0, best_m)
    
    # Pre-extract data for speed
    ops = ready_operations
    durations_list = []
    remaining_works = []
    remaining_ops_list = []
    best_machines = []
    min_durs = []
    
    for i in range(n):
        op = ops[i]
        mo = op["machine_options"]
        durs = list(mo.values())
        min_d = min(durs)
        durations_list.append(durs)
        remaining_works.append(float(op["remaining_work"]))
        remaining_ops_list.append(float(op.get("remaining_ops", 0)))
        # Best machine = min duration
        best_m = min(mo.keys(), key=lambda m: mo[m])
        best_machines.append(best_m)
        min_durs.append(min_d)
    
    # Compute regret for each candidate operation i:
    # If we assign op i to its best machine, it occupies that machine for min_durs[i] time.
    # Other ready ops that share that machine will be delayed by up to min_durs[i].
    # Regret = sum over other ops j that share the same best machine of 
    #          (min_durs[i] * weight_j) where weight_j reflects urgency.
    # We approximate: regret_i = sum_{j != i, best_machines[j] == best_machines[i]} 
    #                          (min_durs[i] / max(1, remaining_works[j]))
    # This penalizes blocking machines that other urgent (low remaining_work) ops need.
    
    # Build machine -> list of op indices mapping
    machine_to_ops = {}
    for i in range(n):
        m = best_machines[i]
        if m not in machine_to_ops:
            machine_to_ops[m] = []
        machine_to_ops[m].append(i)
    
    regrets = [0.0] * n
    for i in range(n):
        m = best_machines[i]
        ops_on_m = machine_to_ops.get(m, [])
        reg = 0.0
        for j in ops_on_m:
            if j == i:
                continue
            # Delay impact: min_durs[i] blocks j for that duration
            # Weight by j's urgency: higher remaining_work = more critical
            urgency_j = remaining_works[j] / max(1.0, min_durs[j])
            reg += min_durs[i] * urgency_j
        regrets[i] = reg
    
    # Composite score: high remaining_work is good, high regret is bad
    # score_i = remaining_works[i] - LAMBDA * regrets[i]
    # Also add small bonus for remaining_ops to keep pipelines flowing
    scores = []
    for i in range(n):
        s = remaining_works[i] - _LAMBDA * regrets[i]
        s += 0.05 * remaining_ops_list[i]
        scores.append(s)
    
    # Pick argmax
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
