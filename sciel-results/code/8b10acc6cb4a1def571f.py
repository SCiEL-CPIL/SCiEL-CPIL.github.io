"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

# Explicit deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time and keep a safety margin.
_START_TIME = _time.perf_counter()
_DEADLINE = 150.0 - 5.0  # 5s safety margin


def _elapsed_ok():
    return (_time.perf_counter() - _START_TIME) < _DEADLINE


def _compute_critical_path_length(op):
    """
    Compute the critical path length for an operation.
    This is the sum of minimum processing times for all remaining operations
    in the job (including this one).
    
    If we have access to remaining_ops and can estimate, we use that.
    Otherwise, we use remaining_work as a proxy.
    """
    machine_options = op["machine_options"]
    remaining_ops = op.get("remaining_ops", 1)
    remaining_work = op.get("remaining_work", sum(machine_options.values()))
    
    # The critical path length is approximately the sum of min durations
    # for all remaining operations. If we only have the current op's machine_options,
    # we estimate based on remaining_work / remaining_ops as average per-op work.
    
    if remaining_ops > 0:
        # Estimate: average min duration per remaining op
        min_duration = min(machine_options.values()) if machine_options else 0
        # The current op contributes min_duration, remaining (remaining_ops - 1) ops
        # contribute approximately remaining_work / remaining_ops each
        if remaining_ops > 1:
            avg_remaining = (remaining_work - min_duration) / (remaining_ops - 1)
            critical_path = min_duration + (remaining_ops - 1) * avg_remaining
        else:
            critical_path = min_duration
    else:
        critical_path = remaining_work
    
    return critical_path


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine runs it.
    
    Strategy: Critical Path Maximization
    - Prioritize the ready operation with the highest remaining critical path length
      (sum of minimum processing times for all subsequent operations in the job)
    - This ensures the bottleneck job is never delayed
    - Among eligible machines, prefer the one with shortest processing time
    
    Returns (index, machine) to explicitly select the machine as well.
    """
    if not _elapsed_ok():
        # Safety: fall back to simple argmax on remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration
        machine_options = op["machine_options"]
        best_machine = min(machine_options, key=lambda m: machine_options[m])
        return (0, best_machine)
    
    # Compute critical path length for each ready operation
    # and track the best (highest critical path)
    
    best_critical_path = -float('inf')
    best_idx = 0
    best_machine = None
    best_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        # Compute critical path length for this operation's job
        critical_path = _compute_critical_path_length(op)
        
        # If this has a higher critical path, it's more urgent
        if critical_path > best_critical_path:
            best_critical_path = critical_path
            best_idx = i
            # Pick the machine with shortest duration for this op
            best_machine = min(machine_options, key=lambda m: machine_options[m])
            best_duration = machine_options[best_machine]
        elif abs(critical_path - best_critical_path) < 1e-9:
            # Tie-breaking: if critical paths are equal, prefer shorter duration
            min_duration = min(machine_options.values())
            if min_duration < best_duration:
                best_idx = i
                best_machine = min(machine_options, key=lambda m: machine_options[m])
                best_duration = min_duration
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
