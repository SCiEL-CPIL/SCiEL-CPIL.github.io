"""Job-Shop Scheduling: Dynamic dispatch rule with SPT and Urgency."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_DEADLINE = 145.0  # Safety margin from 150s limit
_START_TIME = _time.monotonic()

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine to use.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        (index, machine) tuple, where index is the index of the ready operation
        and machine is the selected machine id.
    """
    # Deadline guard
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 2.0:
        if not ready_operations:
            return (0, 0)
        # Fallback: pick first op, min duration machine
        first_op = ready_operations[0]
        mo = first_op["machine_options"]
        if mo:
            best_m = min(mo, key=mo.get)
            return (0, best_m)
        return (0, 0)

    if not ready_operations:
        return (0, 0)

    n = len(ready_operations)
    
    # Pre-allocate arrays
    min_durs = np.empty(n)
    best_machines = np.empty(n, dtype=int)
    rem_ops = np.empty(n)
    rem_work = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            min_durs[i] = np.inf
            best_machines[i] = 0
            rem_ops[i] = 0.0
            rem_work[i] = 0.0
            continue
        
        # Find min duration and corresponding machine
        min_d = np.inf
        best_m = 0
        for m, d in machine_opts.items():
            if d < min_d:
                min_d = d
                best_m = m
        
        min_durs[i] = min_d
        best_machines[i] = best_m
        rem_ops[i] = float(op.get("remaining_ops", 1))
        rem_work[i] = float(op.get("remaining_work", 0))

    # Handle cases where no machine options were valid
    valid_mask = min_durs < np.inf
    if not np.any(valid_mask):
        return (0, 0)

    # Calculate scores
    # Heuristic:
    # 1. Shortest Processing Time (SPT): Prioritize short ops to reduce queue length.
    # 2. Remaining Operations (RO): Prioritize jobs that are close to completion to reduce WIP.
    # 3. Remaining Work: Prioritize jobs with less remaining work to reduce tail latency.
    
    # Weights
    w_spt = 1.0
    w_ro = 0.3
    w_rw = 0.2
    
    scores = np.full(n, -np.inf)
    valid_indices = np.where(valid_mask)[0]
    
    # SPT component: 1 / duration. Higher is better (shorter duration).
    safe_durs = min_durs[valid_indices] + 1e-9
    spt_score = 1.0 / safe_durs
    
    # RO component: 1 / (remaining_ops + 1). Higher is better (fewer ops left).
    safe_ro = rem_ops[valid_indices] + 1.0
    ro_score = 1.0 / safe_ro
    
    # RW component: 1 / (remaining_work + 1). Higher is better (less work left).
    safe_rw = rem_work[valid_indices] + 1.0
    rw_score = 1.0 / safe_rw
    
    # Combined score
    s = w_spt * spt_score + w_ro * ro_score + w_rw * rw_score
    scores[valid_indices] = s
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Get the corresponding best machine
    best_m = best_machines[best_idx]
    
    return (best_idx, best_m)
# EVOLVE-BLOCK-END
