"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a Lexicographic LPT (Longest Processing Time)
    heuristic based on job remaining work, operation duration, and flexibility.
    
    Strategy:
    1. Prioritize jobs with the most remaining work (LPT) to prevent long jobs from becoming
       the critical path tail.
    2. Among jobs with similar remaining work, prioritize operations with the shortest
       processing time on their best machine (SPT) to reduce immediate machine occupancy.
    3. As a final tie-breaker, prioritize operations with fewer machine options (less flexible)
       to secure resources for rigid operations.
    """
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Calculate metrics for each operation
    # 1. Remaining work of the job (higher is better for LPT)
    # 2. Min duration on best machine (lower is better for SPT)
    # 3. Flexibility: number of machines with min duration (lower is better for rigid)
    
    metrics = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_dur = min(durations)
        
        # Count how many machines have the minimum duration (flexibility)
        # If only 1 machine has the min duration, it's less flexible (rigid)
        # If all machines have the same duration, it's highly flexible
        num_min_machines = sum(1 for d in durations if d == min_dur)
        
        remaining_work = op["remaining_work"]
        
        # We want to MAXIMIZE remaining_work
        # We want to MINIMIZE min_dur
        # We want to MINIMIZE num_min_machines (rigid first)
        
        metrics.append({
            'remaining_work': remaining_work,
            'min_dur': min_dur,
            'flexibility': num_min_machines,
            'index': i
        })
    
    # Sort by:
    # 1. remaining_work (Descending) -> Maximize
    # 2. min_dur (Ascending) -> Minimize
    # 3. flexibility (Ascending) -> Minimize
    
    # Python's sort is stable, so we can chain them or use a tuple key.
    # Key: (-remaining_work, min_dur, flexibility)
    # Sorting ascending by this key will put highest remaining_work first (due to negative),
    # then lowest min_dur, then lowest flexibility.
    
    sorted_metrics = sorted(metrics, key=lambda x: (-x['remaining_work'], x['min_dur'], x['flexibility']))
    
    best_idx = sorted_metrics[0]['index']
    
    return best_idx
# EVOLVE-BLOCK-END
