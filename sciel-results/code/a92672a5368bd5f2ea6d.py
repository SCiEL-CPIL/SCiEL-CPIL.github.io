"""Job-Shop Scheduling: Critical Path SPT Dispatch Rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock.
# Keep a safety margin; if we exceed it, fall back to a fast heuristic.
_DEADLINE = 145.0
_START_TIME = _time.monotonic()


def _check_deadline():
    return _time.monotonic() - _START_TIME > _DEADLINE


def _fast_dispatch(ready_operations, time):
    """Fast fallback: shortest processing time first (SPT) on min machine option."""
    if not ready_operations:
        return 0
    durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations.append(min(mo.values()))
        else:
            durations.append(0.0)
    best_idx = int(np.argmin(durations))
    # Return index only for fast path to avoid tuple overhead if simulator is strict, 
    # but the prompt says optionally (index, machine). 
    # The current program returns index. 
    # If I return (index, machine), I need to ensure the simulator handles it.
    # The prompt says "optionally (index, machine)".
    # I will return (index, machine) in the main dispatch to be optimal.
    # For fast dispatch, I'll just return index to be safe and fast.
    return best_idx


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    if _check_deadline():
        return _fast_dispatch(ready_operations, time)

    n = len(ready_operations)
    
    # 1. Identify the maximum remaining work among ready jobs.
    # This approximates the critical path length (lower bound on makespan).
    max_remaining_work = max(op.get("remaining_work", 0.0) for op in ready_operations)
    
    # 2. Filter to operations belonging to jobs with the maximum remaining work.
    # These are the "critical" jobs. Prioritizing them helps reduce the makespan.
    critical_ops = []
    for i, op in enumerate(ready_operations):
        if op.get("remaining_work", 0.0) == max_remaining_work:
            critical_ops.append(i)
    
    # If no critical ops found (should not happen), fall back to all ops.
    if not critical_ops:
        critical_ops = list(range(n))
        
    # 3. Among critical ops, pick the one with the shortest processing time on its best machine.
    # This minimizes the time spent on the critical path and frees the machine sooner.
    best_idx = -1
    best_machine = None
    best_duration = float('inf')
    best_remaining_ops = float('inf')
    
    for i in critical_ops:
        op = ready_operations[i]
        mo = op.get("machine_options", {})
        if not mo:
            continue
            
        # Find the machine with the minimum duration for this operation.
        # This minimizes the time the operation takes.
        min_dur = min(mo.values())
        # Find the corresponding machine.
        # If multiple machines have the same min duration, pick the first one (or any).
        for m, d in mo.items():
            if d == min_dur:
                candidate_machine = m
                break
        
        remaining_ops = op.get("remaining_ops", 0)
        
        # Selection criteria:
        # 1. Shortest duration (min_dur)
        # 2. Fewest remaining operations (to clear jobs)
        # 3. Lowest job id (tie-breaker for stability)
        
        if min_dur < best_duration - 1e-9:
            best_duration = min_dur
            best_idx = i
            best_machine = candidate_machine
            best_remaining_ops = remaining_ops
        elif abs(min_dur - best_duration) < 1e-9:
            if remaining_ops < best_remaining_ops - 1e-9:
                best_remaining_ops = remaining_ops
                best_idx = i
                best_machine = candidate_machine
            elif abs(remaining_ops - best_remaining_ops) < 1e-9:
                if i < best_idx:
                    best_idx = i
                    best_machine = candidate_machine

    if best_idx == -1:
        # Fallback if no valid machine options
        return _fast_dispatch(ready_operations, time)
        
    # Return (index, machine) to explicitly assign the best machine.
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
