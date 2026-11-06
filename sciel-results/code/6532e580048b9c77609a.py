"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We keep a module-level start time and check elapsed time on each dispatch call.
_dispatch_start_time = None
_DEADLINE_SECONDS = 135.0  # safety margin before 150s

def _ensure_deadline():
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()
    return _time.time() - _dispatch_start_time < _DEADLINE_SECONDS


def _estimate_future_machine_load(op, machine_options, ready_operations, time):
    """
    Estimate future load on a machine given current ready operations and the operation's own options.
    We approximate by summing durations of other ready ops that could use the same machine,
    plus the remaining work of the current job on other machines (as a proxy for future contention).
    """
    # Count how many other ready operations also list this machine
    contention = 0
    for other_op in ready_operations:
        if other_op is op:
            continue
        for m in other_op["machine_options"]:
            if m in machine_options:
                # If another ready op can use the same machine, add its min duration as proxy
                contention += min(other_op["machine_options"].values())
                break
    return contention


def dispatch(ready_operations, time):
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()
    
    # If we're close to deadline, fall back to a fast heuristic
    if not _ensure_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    if not ready_operations:
        return 0
    
    # Regret-based heuristic:
    # For each ready operation, compute the "regret" of choosing its best machine over second-best.
    # Regret = (cost_best - cost_second_best) where cost = duration + estimated future machine load
    # Higher regret means the machine choice matters more, so we prioritize those ops.
    # We also blend in urgency (remaining work) to avoid ignoring critical jobs.
    
    n_ops = len(ready_operations)
    regrets = np.zeros(n_ops)
    urgencies = np.zeros(n_ops)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        # Compute cost for each machine option: duration + estimated future load
        costs = []
        for machine, duration in machine_options.items():
            future_load = _estimate_future_machine_load(op, machine_options, ready_operations, time)
            cost = duration + future_load * 0.1  # weight future load less than immediate duration
            costs.append(cost)
        
        if len(costs) >= 2:
            costs_sorted = sorted(costs)
            regret = costs_sorted[1] - costs_sorted[0]  # difference between best and second-best
        elif len(costs) == 1:
            regret = 0.0
        else:
            regret = 0.0
        
        regrets[i] = regret
        urgencies[i] = op["remaining_work"]
    
    # Normalize regrets and urgencies to [0, 1] range for blending
    max_regret = np.max(regrets) if np.max(regrets) > 0 else 1.0
    max_urgency = np.max(urgencies) if np.max(urgencies) > 0 else 1.0
    regrets_norm = regrets / max_regret
    urgencies_norm = urgencies / max_urgency
    
    # Blend: 60% regret, 40% urgency
    scores = 0.6 * regrets_norm + 0.4 * urgencies_norm
    
    best_idx = int(np.argmax(scores))
    
    # Optionally return (index, machine) if the interface supports it.
    # The task says "optionally (index, machine)". Let's just return the index for safety,
    # as the seed program does.
    return best_idx
# EVOLVE-BLOCK-END
