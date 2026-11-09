"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Deadline guard variables
_START_TIME = None
_DEADLINE = 110.0  # Safety margin before 120s limit

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time_module.time()

def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation based on a bottleneck mitigation heuristic.
    Prioritizes operations with high remaining work that can be completed quickly 
    on their least-loaded (shortest duration) machine.
    """
    _init_deadline()
    
    if not ready_operations:
        return 0

    # If we are close to the deadline, return a simple fallback to avoid complex calculation overhead
    if time_module.time() - _START_TIME > _DEADLINE:
        return 0

    best_score = -1
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue

        # Find the machine with the shortest duration for this operation
        # This approximates routing to the "least congested" or fastest resource
        min_duration = float('inf')
        best_m_for_op = None
        
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_m_for_op = m

        if min_duration == float('inf'):
            continue

        # Heuristic Score:
        # High remaining work is good (urgency)
        # Low duration is good (speed/bottleneck mitigation)
        # We use a ratio to balance them. Adding 1 to denominator to avoid div by zero.
        # We also add a small weight to remaining_ops to break ties in favor of jobs with more steps left
        remaining_ops = op.get("remaining_ops", 0)
        
        # Score formulation:
        # Primary: Work-to-Time ratio
        score = remaining_work / (min_duration + 1.0)
        
        # Secondary: Slight bonus for jobs with more remaining ops (longer path)
        score += 0.1 * remaining_ops

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Return the index and the selected machine
    # The interface allows returning (index, machine)
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
