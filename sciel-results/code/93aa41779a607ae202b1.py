"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 115.0  # safety margin before 120s kill
_start_time = _time.time()


def _elapsed():
    return _time.time() - _start_time


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: Constraint tightness (fewest machine options) as primary key,
    then remaining work as secondary, then shortest processing time on best machine.
    Returns (index, machine) tuple to also select the machine.
    """
    if not ready_operations:
        return 0

    if _elapsed() > _DEADLINE:
        # Safety: just pick the one with most remaining work quickly
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    
    # Compute scores for each ready operation
    best_idx = 0
    best_score = None
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)
        
        # Number of machine options (constraint tightness)
        num_options = len(machine_options)
        
        # Shortest processing time among options
        shortest_pt = min(machine_options.values())
        
        # Longest processing time among options
        longest_pt = max(machine_options.values())
        
        # Average processing time
        avg_pt = sum(machine_options.values()) / num_options
        
        # Constraint tightness: fewer options = more constrained = higher priority
        # Use inverse of num_options as a priority factor
        tightness = 1.0 / (num_options + 1)  # +1 to avoid issues
        
        # Work remaining factor
        work_factor = remaining_work
        
        # Shortest processing time factor (prefer shorter to reduce makespan)
        pt_factor = shortest_pt
        
        # Remaining ops factor (jobs with more remaining ops might need careful scheduling)
        ops_factor = remaining_ops
        
        # Composite score: we want to maximize this
        # Primary: constraint tightness (fewer options -> higher priority)
        # Secondary: more remaining work
        # Tertiary: shorter processing time
        # Quaternary: fewer remaining ops (to finish jobs sooner)
        
        score = (
            tightness * 1000.0 +  # primary: constraint tightness
            work_factor * 1.0 +    # secondary: remaining work
            -pt_factor * 0.1 +     # tertiary: prefer shorter PT
            -ops_factor * 0.5      # quaternary: prefer fewer remaining ops
        )
        
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
            # Select the machine with the shortest processing time for this operation
            best_machine = min(machine_options, key=machine_options.get)
    
    # Return (index, machine) to also specify which machine to use
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
