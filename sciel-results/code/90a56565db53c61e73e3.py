"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Machine-load balancing strategy:
    Select the ready operation with the minimum remaining workload on its most
    constrained machine (the machine with the fewest available options or highest
    current load proxy), thereby preventing any single machine from becoming the
    dominant bottleneck.
    
    Heuristic:
    1. For each ready operation, identify the 'most constrained' machine among
       its machine_options. Constraint proxy = number of other ready ops also
       targeting that machine (higher = more constrained).
    2. Score each operation: prefer operations that go to highly constrained
       machines with short durations (to clear the bottleneck quickly).
    3. Tie-break by total remaining work (less remaining = finish sooner).
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Count how many ready operations are targeting each machine
    machine_load = defaultdict(int)
    for op in ready_operations:
        for machine in op["machine_options"]:
            machine_load[machine] += 1
    
    # For each operation, compute a score
    # We want to pick the operation that:
    # - Targets the most constrained machine (high machine_load)
    # - Has a short duration on that machine (to clear it fast)
    # - Has less total remaining work (to finish sooner)
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        # Find the most constrained machine for this operation
        # Among machines this op can run on, pick the one with highest load
        if not machine_options:
            continue
        
        # Get the machine with highest load among options
        max_load = max(machine_load[m] for m in machine_options)
        
        # Among machines with max load, pick the one with shortest duration
        best_duration = min(machine_options[m] for m in machine_options if machine_load[m] == max_load)
        
        # Score: high load is good (clears bottleneck), short duration is good,
        # low remaining work is good
        # Normalize: score = load * (1/duration) - remaining_work * small_factor
        # Actually, let's think more carefully:
        # We want to prioritize operations that will reduce the load on the most
        # constrained machines quickly.
        
        # Primary: target the most loaded machine
        # Secondary: shortest processing time on that machine
        # Tertiary: least remaining work overall
        
        score = max_load * 100.0 - best_duration * 10.0 - remaining_work * 0.1
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
