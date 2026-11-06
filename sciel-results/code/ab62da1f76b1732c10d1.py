"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Uses a composite heuristic that considers:
    1. Minimum processing time (SPT-like) to reduce blocking
    2. Maximum remaining work (to prioritize critical jobs)
    3. Machine load balancing (prefer machines with fewer options/less contention)
    
    Returns: (index, machine) tuple where index is the ready operation to start
             and machine is the chosen machine from its options.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # For each ready operation, compute a score
    # We want to pick the operation that balances urgency and efficiency
    
    scores = np.zeros(n)
    chosen_machines = [None] * n
    
    # Precompute some statistics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration option (SPT) and consider machine load
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores[i] = -np.inf
            continue
        
        # Find the machine with minimum processing time (SPT heuristic)
        min_duration = float('inf')
        best_machine = None
        for machine, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                best_machine = machine
        
        chosen_machines[i] = best_machine
        
        # Composite score:
        # - Higher remaining work = more urgent (positive contribution)
        # - Lower minimum duration = less blocking (positive contribution)
        # - Fewer machine options = less flexibility, may be more constrained
        
        # Normalize remaining work contribution
        rw_score = remaining_works[i]
        
        # SPT contribution: prefer shorter processing times
        spt_score = 1.0 / (min_duration + 1e-6)
        
        # Combine: weight remaining work and SPT
        # Use a ratio to balance the two scales
        scores[i] = rw_score * 0.5 + spt_score * 0.5
    
    # Pick the operation with the highest composite score
    best_idx = int(np.argmax(scores))
    
    # Return (index, machine) to allow flexible machine selection
    return (best_idx, chosen_machines[best_idx])
# EVOLVE-BLOCK-END
