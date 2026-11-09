"""Job-Shop Scheduling Dispatch Rule.
Uses a heuristic combining Criticality, SPT efficiency, Urgency, and Bottleneck.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 120s/150s limit

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Deadline check: if we are running out of time, use a simple fallback
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(float('inf'))
        idx = int(np.argmin(min_durs))
        op = ready_operations[idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (idx, best_machine)
        return idx

    # Extract features
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op.get("remaining_ops", 1) for op in ready_operations], dtype=np.float64)
    
    hard_min_durations = np.zeros(n, dtype=np.float64)
    num_options = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            durs = list(machine_options.values())
            best_m, min_d = min(machine_options.items(), key=lambda item: item[1])
            hard_min_durations[i] = min_d
            num_options[i] = len(machine_options)
            best_machines[i] = best_m
        else:
            hard_min_durations[i] = 1.0
            num_options[i] = 1.0
            best_machines[i] = None

    # Avoid division by zero
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)
    hard_min_durations_safe = hard_min_durations + 1e-9

    # Normalization
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    dur_min = np.min(hard_min_durations_safe)
    dur_max = np.max(hard_min_durations_safe)
    
    # SPT: shorter processing time is better (1/duration)
    if dur_max > dur_min:
        norm_dur = (dur_max - hard_min_durations_safe) / (dur_max - dur_min)
    else:
        norm_dur = np.ones(n)
    
    # Urgency: fewer remaining ops is better
    max_ops = np.max(remaining_ops_safe) if np.max(remaining_ops_safe) > 0 else 1.0
    norm_ops = (max_ops - remaining_ops_safe) / max_ops
    
    # Criticality: more remaining work is better
    norm_work = remaining_works / max_work

    # Bottleneck: fewer machine options means more constrained (higher priority)
    max_opts = np.max(num_options) if np.max(num_options) > 0 else 1.0
    min_opts = np.min(num_options)
    if max_opts > min_opts:
        norm_bottleneck = (max_opts - num_options) / (max_opts - min_opts)
    else:
        norm_bottleneck = np.ones(n)

    # Weighted Sum
    # SPT and bottleneck terms balance local efficiency with global urgency
    w_work = 0.30
    w_dur = 0.35
    w_ops = 0.15
    w_bottleneck = 0.20
    
    scores = (w_work * norm_work) + (w_dur * norm_dur) + (w_ops * norm_ops) + (w_bottleneck * norm_bottleneck)
    
    best_idx = int(np.argmax(scores))
    
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END
