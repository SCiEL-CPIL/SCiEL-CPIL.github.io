"""Open-shop dispatch rule: Load-Balancing Water-Filling with L-infinity norm scoring."""
# EVOLVE-BLOCK-START
import time
import math

# Global time budget tracking
_START_TIME = time.time()
# Evaluator kills at 150s wall-clock. We reserve safety margin.
_DEADLINE = _START_TIME + 145.0

# State tracking for load balancing
_machine_total_load = {}

def _time_left():
    return _DEADLINE - time.time()


def _get_max_load(machine_loads):
    """Get the current maximum load across all machines."""
    if not machine_loads:
        return 0.0
    return max(machine_loads.values())


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Select the job with Most Work Remaining (MWR).
    2. For that job's ready operation, evaluate each machine option by simulating
       the assignment and scoring by the resulting L-infinity norm (max machine load).
    3. Prefer assignments that minimally increase the global maximum load.
    4. Return (index, machine) to explicitly assign the machine.
    """
    if not ready_operations:
        return 0
    
    # Time guard: if we're running low on time, use a simple fallback
    if _time_left() < 0.5:
        max_remaining = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
                best_idx = i
        return best_idx

    # 1. Find the job with maximum remaining work (MWR)
    job_remaining = {}
    for op in ready_operations:
        job_id = op["job"]
        rw = op.get("remaining_work", 0)
        if job_id not in job_remaining or rw > job_remaining[job_id]:
            job_remaining[job_id] = rw
    
    if not job_remaining:
        return 0
    
    max_job_remaining = max(job_remaining.values())
    
    # Find candidates: ops belonging to jobs with max remaining work
    candidates = []
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        rw = op.get("remaining_work", 0)
        if rw == max_job_remaining:
            candidates.append((i, op))
            
    if not candidates:
        return 0

    # 2. Among candidates, pick the best (index, machine) pair
    # Score: resulting max machine load (L-infinity norm) after assignment
    # Tie-breaker: minimize the assigned machine's new load, then prefer shorter duration
    
    current_max_load = _get_max_load(_machine_total_load)
    
    best_idx = -1
    best_machine = None
    best_max_load = float('inf')
    best_assigned_load = float('inf')
    best_duration = float('inf')
    
    for i, op in candidates:
        machine_options = op["machine_options"]
        for m, d in machine_options.items():
            current_load = _machine_total_load.get(m, 0.0)
            new_assigned_load = current_load + d
            
            # Simulate: what would the max load be if we assigned this?
            # The max is either the current max (if another machine already has it)
            # or the new assigned load (if this machine becomes the bottleneck)
            resulting_max = max(current_max_load, new_assigned_load)
            
            # Primary: minimize resulting max load (L-infinity)
            # Secondary: minimize the assigned machine's load (local balance)
            # Tertiary: prefer shorter duration (SPT)
            if (resulting_max < best_max_load or
                (resulting_max == best_max_load and new_assigned_load < best_assigned_load) or
                (resulting_max == best_max_load and new_assigned_load == best_assigned_load and d < best_duration)):
                best_max_load = resulting_max
                best_assigned_load = new_assigned_load
                best_duration = d
                best_idx = i
                best_machine = m
                
    if best_idx == -1:
        # Fallback
        return 0

    # 3. Update global state
    _machine_total_load[best_machine] = best_assigned_load
    
    # Return tuple to specify machine assignment
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
