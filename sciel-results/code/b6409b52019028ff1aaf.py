"""Open-Shop Scheduling dispatch rule using a structural change:
   machine-utilization-aware selection with dynamic contention and
   job-criticality weighting."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = 150.0
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
        return (0, 0)
    
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick op with max remaining work, assign to min-duration machine
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

    # --- Structural change: machine-load-aware scoring ---
    # 1. Compute for each machine: total work already assigned (proxy via contention * avg duration)
    # 2. For each operation, compute a "urgency" score based on:
    #    - remaining_work (criticality)
    #    - remaining_ops (how many more ops this job has)
    #    - bottleneck potential (max duration across machines)
    # 3. For machine assignment, use a load-aware cost:
    #    cost(m) = duration * (1 + alpha * load_factor(m))
    #    where load_factor reflects how "busy" the machine is relative to others
    
    # Compute machine contention (number of ready ops that can use each machine)
    machine_contention = {}
    machine_total_duration = {}  # sum of durations over all ready ops for each machine
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_contention[m] = machine_contention.get(m, 0) + 1
            machine_total_duration[m] = machine_total_duration.get(m, 0) + d
    
    # Normalize machine loads: a machine with high total duration is "busier"
    all_machines = set(machine_total_duration.keys())
    if all_machines:
        max_total = max(machine_total_duration.values())
        min_total = min(machine_total_duration.values())
        range_total = max_total - min_total if max_total > min_total else 1.0
    else:
        max_total = 1.0
        min_total = 0.0
        range_total = 1.0
    
    # Load factor: normalized to [0, 1]
    machine_load_factor = {}
    for m in all_machines:
        load = (machine_total_duration[m] - min_total) / range_total
        machine_load_factor[m] = load
    
    # Parameters
    W_RW = 1.0        # weight of remaining work
    W_REM = 0.3       # weight of remaining ops
    W_MAX = 0.5       # weight of max duration
    W_MIN = 0.3       # weight of min duration (prefer shorter for less blocking)
    W_COST = 1.2      # weight of machine cost penalty
    ALPHA = 0.8       # weight of load factor in cost
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        rem_ops = float(op.get("remaining_ops", 1))
        durations = op["machine_options"]
        
        if not durations:
            continue

        durs = list(durations.values())
        max_dur = max(durs)
        min_dur = min(durs)
        avg_dur = sum(durs) / len(durs)
        
        # Compute cost for each machine: duration * (1 + alpha * load_factor)
        costs = {}
        for m, d in durations.items():
            lf = machine_load_factor.get(m, 0.0)
            cont = machine_contention.get(m, 0)
            # Combine load factor and contention
            effective_load = 0.6 * lf + 0.4 * (cont / max(n, 1))
            costs[m] = d * (1.0 + ALPHA * effective_load)
        
        # Find the best machine (min cost)
        best_m_for_op = min(costs, key=lambda m: costs[m])
        min_cost_val = costs[best_m_for_op]
        
        # Urgency score: high rw, high rem_ops, high max_dur (bottleneck), low min_cost
        score = (W_RW * rw + W_REM * rem_ops + W_MAX * max_dur + W_MIN * min_dur) \
                - W_COST * min_cost_val
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
