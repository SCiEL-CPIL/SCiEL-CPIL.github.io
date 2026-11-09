"""Open-Shop Scheduling: load-aware dispatch balancing machine congestion and job urgency."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Module-level state for deadline guard and machine load tracking
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 15s safety margin before 160s kill (task says 120s eval, but guard for 150s)

# Track machine loads across calls within an instance
_machine_load = {}
# Track which jobs are still active
_active_jobs = set()


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Load-aware dispatch that balances:
    1. Machine congestion (avoid assigning to already-busy machines)
    2. Job urgency (jobs with more remaining operations should be prioritized)
    3. Work remaining (larger remaining work gets priority)
    
    For each ready operation, we consider the best (minimum duration) machine option,
    but penalize machines that are already heavily loaded. We also weight by the
    job's remaining operations count to prioritize jobs that are close to completion
    or have many ops left (to avoid stragglers).
    """
    global _machine_load, _active_jobs, _START_TIME, _DEADLINE
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Track active jobs
    for op in ready_operations:
        _active_jobs.add(op["job"])
    
    # For each ready operation, compute a priority score
    # We want to MINIMIZE the score (pick the best operation)
    # Score components:
    #   - estimated completion time if we assign to the least-loaded compatible machine
    #   - urgency: jobs with more remaining ops should go first (to avoid long tails)
    #   - work remaining: larger work means it blocks the machine longer
    
    best_idx = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        
        if not machine_options:
            continue
        
        # Find the machine with minimum (current_load + duration) for this op
        # This approximates when the operation would finish if assigned now
        min_completion = float('inf')
        for machine, duration in machine_options.items():
            current_load = _machine_load.get(machine, 0.0)
            completion = current_load + duration
            if completion < min_completion:
                min_completion = completion
        
        # Urgency factor: jobs with more remaining ops should be prioritized
        # This helps complete jobs sooner, freeing up resources
        # Use remaining_ops as a weight; also consider remaining_work
        # A job with many remaining ops is "urgent" because delaying it delays everything after
        
        # Combine: we want operations that can finish early AND belong to urgent jobs
        # Score = min_completion + urgency_penalty
        # Higher remaining_ops means more urgency (more ops to do after this one)
        # But we also want to account for the fact that doing a big op early might be good
        
        # Let's use: score = min_completion * (1 + 0.1 * remaining_ops) 
        # This means if an op finishes at time 10 and the job has 3 remaining ops,
        # effective score is 10 * 1.3 = 13, vs an op finishing at time 11 with 1 remaining op: 11 * 1.1 = 12.1
        # So the earlier-finishing op with more remaining work wins.
        
        urgency_multiplier = 1.0 + 0.15 * remaining_ops
        
        score = min_completion * urgency_multiplier
        
        # Small tiebreaker: prefer larger remaining_work (to get big jobs moving)
        score -= 0.01 * remaining_work
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
