"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy:
    1. For each ready operation, select the best machine (shortest duration, tie-break by machine id for determinism).
    2. Score each operation using a composite heuristic:
       - Primary: remaining_work (criticality) - higher is better
       - Secondary: remaining_ops - fewer remaining ops means less tail latency
       - Tertiary: shortest processing time on chosen machine (SPT) to reduce blocking
       
    The score is designed to prioritize operations that:
    - Have the most remaining work (critical path)
    - Have fewer remaining operations (to finish jobs sooner and free up resources)
    - Can be processed quickly on their chosen machine (SPT)
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # For each operation, find the best machine (shortest duration)
    best_machine_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            # Find machine with shortest duration
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_duration = machine_options[best_machine]
        else:
            best_duration = 0
        best_machine_durations.append(best_duration)
    
    # Extract features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    
    # Normalize features for combination
    max_work = max(remaining_works) if remaining_works else 1
    min_work = min(remaining_works) if remaining_works else 0
    work_range = max_work - min_work if max_work > min_work else 1
    
    max_ops = max(remaining_ops) if remaining_ops else 1
    min_ops = min(remaining_ops) if remaining_ops else 0
    ops_range = max_ops - min_ops if max_ops > min_ops else 1
    
    max_duration = max(best_machine_durations) if best_machine_durations else 1
    min_duration = min(best_machine_durations) if best_machine_durations else 0
    duration_range = max_duration - min_duration if max_duration > min_duration else 1
    
    # Composite score:
    # - Higher remaining work is better (criticality)
    # - Fewer remaining ops is better (less tail)
    # - Shorter duration on best machine is better (SPT)
    
    best_idx = 0
    best_score = float('-inf')
    
    for i in range(n):
        # Normalize to [0, 1]
        work_norm = (remaining_works[i] - min_work) / work_range
        ops_norm = (max_ops - remaining_ops[i]) / ops_range  # Invert: fewer ops = higher
        duration_norm = (max_duration - best_machine_durations[i]) / duration_range  # Invert: shorter = higher
        
        # Weighted combination: criticality most important, then tail, then SPT
        score = 0.5 * work_norm + 0.3 * ops_norm + 0.2 * duration_norm
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
