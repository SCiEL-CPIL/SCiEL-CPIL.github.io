"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random

# Track elapsed time across all dispatch calls to respect the 150s wall-clock limit.
# Since dispatch is called many times, we need to be very fast per call.
# We'll use a simple heuristic that's O(n) where n is the number of ready operations.

_START_TIME = _time.monotonic()
_DEADLINE = 145.0  # Safety margin of 5 seconds before the 150s hard limit


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: Use a composite heuristic that considers:
    1. Remaining work (more work first - priority rule)
    2. Number of remaining operations (more ops first - to keep jobs flowing)
    3. Minimum processing time across available machines (shortest processing time)
    4. Machine load balancing (prefer less loaded machines)
    
    Returns index of the ready operation to start, optionally with machine choice.
    """
    # Deadline guard: if we're close to the limit, just return the first operation
    if _time.monotonic() - _START_TIME > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        # If only one operation, pick it and the best machine
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0
    
    # Composite heuristic scoring
    # For each ready operation, compute a score. Higher score = more urgent/priority.
    
    # Precompute some global stats if possible
    # We want to favor:
    # - Operations with more remaining work (MWL - Most Work Left)
    # - Operations with more remaining operations (to reduce idle time)
    # - Operations with shorter minimum processing time (SPT - Shortest Processing Time)
    # - Operations on machines that are less loaded (load balancing)
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            scores.append(0.0)
            best_machines.append(None)
            continue
        
        # Minimum processing time across available machines
        min_proc_time = min(machine_options.values())
        
        # Choose the machine with the shortest processing time for this operation
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        best_machines.append(best_machine)
        
        # Composite score:
        # - remaining_work: prioritize jobs with more total remaining work
        # - remaining_ops: prioritize jobs with more operations left (to keep them flowing)
        # - 1/min_proc_time: prioritize shorter processing times (SPT)
        # We normalize these to avoid scale issues
        
        score = 0.0
        if remaining_work > 0:
            score += remaining_work / max(remaining_work, 1)
        if remaining_ops > 0:
            score += remaining_ops / max(remaining_ops, 1)
        if min_proc_time > 0:
            score += 1.0 / min_proc_time
        
        scores.append(score)
    
    # Find the index with the highest score
    max_score = max(scores)
    best_idx = scores.index(max_score)
    
    # If we have a best machine choice, return (index, machine)
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    
    return best_idx
# EVOLVE-BLOCK-END
