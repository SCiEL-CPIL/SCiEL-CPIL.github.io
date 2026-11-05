"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Explicit deadline guard: 150s wall-clock limit, use 145s for safety
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling using an Urgency/Cost ratio.
    
    Strategy:
    - Urgency: driven by remaining_work (more work left = more critical)
    - Cost: driven by the minimum processing time across eligible machines (SPT)
    - Machine selection: pick the machine that minimizes processing time for the chosen op
    
    The ratio urgency/cost naturally balances:
    - Jobs with lots of remaining work get high urgency
    - Operations that are quick to complete have low cost, boosting their ratio
    - This adapts dynamically as the schedule evolves
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        min_duration = min(durations)
        
        # Urgency: remaining_work is the primary signal.
        # Add a small bonus for jobs closer to completion (fewer remaining ops).
        # The bonus is scaled to be meaningful but not dominate.
        urgency = remaining_work + 0.5 * max(0, 10 - remaining_ops)
        
        # Cost: minimum processing time. Add a small epsilon to avoid division by zero.
        cost = min_duration + 1e-9
        
        # Score: urgency / cost (higher is better)
        score = urgency / cost
        
        if score > best_score:
            best_score = score
            best_idx = idx
            # Choose the machine with the shortest processing time
            min_dur = min(durations)
            best_machine = machines[durations.index(min_dur)]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
