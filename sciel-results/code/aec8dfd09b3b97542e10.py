"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 5.0  # return before this margin


def _time_left():
    return _DEADLINE - _time.time() - _SAFETY_MARGIN


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: Combine multiple heuristics to pick the best (operation, machine) pair.
    - Prioritize operations with more remaining work (to finish long jobs early)
    - Prefer machines with shorter processing times (to reduce makespan)
    - Consider the earliest available time on each machine
    - Use a weighted combination of these factors
    
    Returns: index of the ready operation to start, or (index, machine) tuple
    """
    if not ready_operations:
        return 0
    
    # Safety check
    if _time_left() < 0.1:
        # Fallback: just pick the one with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    n_ops = len(ready_operations)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # No machine options, skip or use default
            continue
        
        # Find the best machine for this operation
        min_duration = float('inf')
        best_machine_for_op = None
        
        for machine, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                best_machine_for_op = machine
        
        if best_machine_for_op is None:
            continue
        
        # Compute a composite score
        # 1. Remaining work (higher is better for prioritizing)
        work_score = remaining_work
        
        # 2. Shorter processing time is better (lower makespan)
        # Normalize by considering the ratio of min duration to remaining work
        if remaining_work > 0:
            time_ratio = min_duration / remaining_work
        else:
            time_ratio = 1.0
        
        # 3. Fewer remaining operations might mean this is closer to completion
        #    but we want to start operations that have more work left to avoid idle time
        ops_score = remaining_ops
        
        # Composite: prioritize high remaining work, short processing time
        # Use a weighted sum with emphasis on work and processing time
        score = 1.0 * work_score - 0.5 * min_duration + 0.1 * ops_score
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    # If we found a best machine, return the tuple (index, machine)
    # Otherwise, just return the index
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
