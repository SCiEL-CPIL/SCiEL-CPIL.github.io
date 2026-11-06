"""Improved dispatch rule for flexible job-shop scheduling.
Strategy: WSJF-like ratio heuristic with explicit machine selection.
We return (index, machine) to let the simulator use our machine choice.
"""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard
_START_TIME = _time.time()
_DEADLINE_SECONDS = 110.0


def _elapsed():
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """Pick which ready operation to start next and which machine to use.
    
    Returns (index, machine) where index is the index into ready_operations
    and machine is the chosen machine for that operation.
    """
    if not ready_operations:
        return (0, None)

    n = len(ready_operations)

    # Deadline guard: if close to kill, use a very cheap rule
    if _elapsed() > _DEADLINE_SECONDS:
        # Pick operation with most remaining work, assign to fastest machine
        best_idx = 0
        best_rw = ready_operations[0].get("remaining_work", 0)
        for i in range(1, n):
            rw = ready_operations[i].get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        op = ready_operations[best_idx]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
        else:
            best_m = None
        return (best_idx, best_m)

    # Main heuristic: WSJF-like ratio
    # For each operation, compute:
    #   remaining_ops / min_duration  (lower is better -> more urgent per unit time)
    #   Also consider remaining_work as a tiebreaker
    #   Machine selection: pick machine with minimum duration (fastest)
    #   Additional consideration: if we can estimate machine load, prefer less loaded
    
    best_score = -1e18
    best_idx = 0
    best_machine = None

    for i in range(n):
        op = ready_operations[i]
        mo = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", 0)

        if not mo:
            continue

        # Find the fastest machine (minimum duration)
        min_dur = float('inf')
        fastest_m = None
        for m, dur in mo.items():
            if dur < min_dur:
                min_dur = dur
                fastest_m = m

        if min_dur == float('inf'):
            continue

        # WSJF-like ratio: remaining_ops / min_duration
        # Lower ratio = more urgent (fewer remaining ops relative to how fast we can process)
        # Actually, we want to prioritize operations that are "critical" - 
        # high remaining work relative to time, or few remaining ops (near end of job)
        
        # Let's think about this differently:
        # - remaining_ops: how many more operations this job has (including current?)
        # - remaining_work: total remaining processing time for this job
        # - We want to prioritize jobs that are close to completion (few remaining ops)
        #   AND jobs with high remaining work (important jobs)
        
        # A good heuristic: prioritize by remaining_work / remaining_ops (avg work per op)
        # combined with urgency (1/remaining_ops)
        
        # Let's use a composite score:
        # score = remaining_work * (1.0 / (1.0 + remaining_ops)) / min_dur
        # This rewards: high remaining work, few remaining ops, fast processing
        
        if remaining_ops is None or remaining_ops < 1:
            remaining_ops = 1
            
        # Weighted combination:
        # Primary: remaining_work (more work = more important)
        # Secondary: urgency (fewer remaining ops = more urgent)  
        # Tertiary: speed (faster machine = better)
        
        urgency = 1.0 / (1.0 + remaining_ops)  # Higher when fewer remaining ops
        work_factor = remaining_work if remaining_work is not None else 0.0
        speed_factor = 1.0 / min_dur if min_dur > 0 else 0.0
        
        # Composite score
        score = 0.6 * work_factor * urgency + 0.4 * speed_factor * (1.0 + urgency)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = fastest_m

    # Fallback if no valid operation found
    if best_machine is None:
        best_idx = 0
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo.keys(), key=lambda m: mo[m])
        else:
            best_machine = None

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
