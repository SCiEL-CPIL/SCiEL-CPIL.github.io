"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

# Deadline guard: track elapsed time to ensure we stay well within the 150s limit
_start_time = _time.time()
_DEADLINE = 145.0  # 5s margin before 150s kill

# Track state to detect schedule progression
_last_time = 0.0
_call_count = 0


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Use a composite heuristic that considers:
    1. Machine load balancing (prefer operations on less-loaded machines)
    2. Shortest processing time on the chosen machine
    3. Most work remaining to keep busy jobs progressing
    4. Fewest remaining operations to complete jobs sooner
    
    The key structural change: instead of just picking by a static weighted sum,
    we consider the actual machine assignment implied by each operation and
    prioritize based on which machine would be least congested.
    """
    global _last_time, _call_count
    _call_count += 1
    
    if not ready_operations:
        return 0
    
    # Deadline guard: if we're running out of time, use a fast fallback
    if _time.time() - _start_time > _DEADLINE:
        # Fast fallback: shortest processing time
        min_duration = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            min_opt = min(op["machine_options"].values())
            if min_opt < min_duration:
                min_duration = min_opt
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # For each ready operation, find the best machine option
    # and compute a composite score
    best_score = -float('inf')
    best_idx = 0
    
    # Pre-compute some statistics
    total_remaining_work = sum(op["remaining_work"] for op in ready_operations)
    total_remaining_ops = sum(op["remaining_ops"] for op in ready_operations)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        
        # Find the machine with the shortest duration for this operation
        min_dur = float('inf')
        best_machine = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_machine = m
        
        # Score components:
        # 1. SPT component: shorter is better
        # Normalize by total work to make it scale-invariant
        spt_component = 1.0 / (min_dur + 1e-9)
        
        # 2. Work remaining component: more work remaining means the job is more critical
        #    (if we don't start it, it may block later)
        mwr_component = rw / (total_remaining_work + 1e-9)
        
        # 3. Operations remaining: fewer remaining ops means we can finish the job sooner
        #    which frees up resources
        ops_component = 1.0 / (ro + 1e-9)
        
        # 4. Machine congestion heuristic: prefer operations that can run on a machine
        #    that has fewer other ready operations competing for it
        #    Count how many other ready operations also want this machine
        machine_competition = 0
        for j, other_op in enumerate(ready_operations):
            if j != i and best_machine in other_op["machine_options"]:
                machine_competition += 1
        
        # Lower competition is better
        competition_component = 1.0 / (machine_competition + 1.0)
        
        # Composite score with adaptive weights
        # Emphasize SPT and low competition more heavily
        w_spt = 0.35
        w_mwr = 0.25
        w_ops = 0.20
        w_comp = 0.20
        
        score = (w_spt * spt_component + 
                 w_mwr * mwr_component + 
                 w_ops * ops_component + 
                 w_comp * competition_component)
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
