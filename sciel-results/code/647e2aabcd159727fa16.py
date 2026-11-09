"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: ensure we finish well within the 120s/150s limits.
_DEADLINE = 115.0  # seconds from process start; safety margin
_START_TIME = _time.monotonic()


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Strategy: Dynamic bottleneck optimization.
    For each ready operation, compute:
      - job_critical_path: sum of min durations of remaining ops in the job
      - machine_load_ratio: for each eligible machine, estimate its congestion
        by summing min durations of all ops that could use that machine
        (we approximate this using the ready_operations set and remaining_work)
    
    We score each operation by:
      score = job_critical_path / (1 + estimated_machine_congestion)
    
    We also select the machine with the lowest estimated congestion for that op.
    
    This prioritizes operations on jobs with long remaining critical paths
    while avoiding machines that appear heavily loaded.
    """
    # Safety check: if we're close to deadline, fall back to simple heuristic
    if _time.monotonic() - _START_TIME > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    
    if n_ops == 1:
        op = ready_operations[0]
        # Pick fastest machine
        machine_options = op["machine_options"]
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0
    
    # Estimate machine congestion: for each machine, sum the min duration
    # of all ready operations that can use it. This is a proxy for load.
    # We also incorporate remaining_work as a signal of how much work is queued.
    
    machine_min_load = {}  # machine -> sum of min durations of ready ops using it
    
    for op in ready_operations:
        mo = op["machine_options"]
        for m in mo:
            if m not in machine_min_load:
                machine_min_load[m] = 0
            machine_min_load[m] += mo[m]
    
    # For each ready operation, compute a score
    best_score = -1
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            continue
        
        # Job critical path approximation: use remaining_work as proxy
        # (remaining_work likely sums remaining op durations)
        job_cp = op["remaining_work"]
        
        # Find the machine with lowest congestion for this op
        min_congestion = float('inf')
        best_m_for_op = None
        for m, dur in mo.items():
            congestion = machine_min_load.get(m, 0)
            # Weight by duration on this machine
            effective_load = congestion + dur
            if effective_load < min_congestion:
                min_congestion = effective_load
                best_m_for_op = m
        
        if best_m_for_op is None:
            best_m_for_op = min(mo.keys(), key=lambda m: mo[m])
            min_congestion = machine_min_load.get(best_m_for_op, 0) + mo[best_m_for_op]
        
        # Score: higher is better
        # We want high critical path, low machine congestion
        score = job_cp / (1.0 + min_congestion)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
