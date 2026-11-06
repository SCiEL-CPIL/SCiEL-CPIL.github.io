"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    """Check if we are approaching the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    # Explicit deadline guard
    if _time_exceeded():
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0
        
    # Pre-calculate max remaining work for normalization
    max_rw = 0.0
    for op in ready_operations:
        rw = op.get("remaining_work", 0.0)
        if rw > max_rw:
            max_rw = rw
    if max_rw == 0:
        max_rw = 1.0

    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        # Check deadline again
        if _time_exceeded():
            break

        remaining_work = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            # Fallback if no machines (should not happen)
            score = 0.0
            best_machine_local = 0
        else:
            # Find the machine with the minimum duration
            min_duration = float('inf')
            best_machine_local = next(iter(machine_options.keys()))
            
            for m, dur in machine_options.items():
                if dur < min_duration:
                    min_duration = dur
                    best_machine_local = m
            
            # Avoid division by zero
            if min_duration <= 0:
                min_duration = 1e-6
            
            # Heuristic Components:
            # 1. SPT (Shortest Processing Time): Strongly favors short ops to free machines.
            spt_score = 1.0 / (min_duration + 1.0)
            
            # 2. Work Remaining: Favors jobs with more work left (Criticality).
            # Normalized to [0, 1] based on current batch max.
            work_score = remaining_work / max_rw
            
            # 3. Tail Bonus: Favors jobs close to completion to reduce active jobs.
            # Inverse of remaining ops count.
            if remaining_ops > 0:
                tail_score = 2.0 / remaining_ops
            else:
                tail_score = 2.0

            # Combined Score:
            # We prioritize SPT heavily, then Work, then Tail.
            # Weights can be tuned. SPT is usually the strongest for makespan.
            score = 5.0 * spt_score + 1.0 * work_score + tail_score

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_local
            
    # Return the index and the explicitly selected best machine
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
