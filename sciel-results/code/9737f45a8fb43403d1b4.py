"""Open-shop dispatch rule: Combined MWR job selection with SPT machine assignment."""
# EVOLVE-BLOCK-START
import time
import random

# Global time budget tracking
_START_TIME = time.time()
# Evaluator kills at 150s wall-clock. We reserve safety margin.
_DEADLINE = _START_TIME + 145.0


def _time_left():
    return _DEADLINE - time.time()


def _best_machine_for_op(op, machine_loads):
    """
    For a given operation, find the best machine to assign.
    Uses a combination of:
    1. Shortest processing time (SPT) preference - assign to machine where this op is shortest
    2. Load balancing - prefer machines with less current load
    3. Consider remaining work on other machines for this job
    
    Returns (machine, duration)
    """
    machine_options = op["machine_options"]
    best_machine = None
    best_score = float('inf')
    
    for machine, duration in machine_options.items():
        # Score: lower is better
        # Component 1: duration (SPT - shorter is better)
        # Component 2: current load on that machine (less loaded is better)
        # Component 3: slight preference for machines where this job has less remaining work elsewhere
        
        load = machine_loads.get(machine, 0)
        
        # Normalize-ish: weight duration heavily, then load
        # We want to minimize makespan, so:
        # - Assign short ops to any available machine (SPT)
        # - Balance load across machines
        score = duration * 1.0 + load * 0.3
        
        if score < best_score:
            best_score = score
            best_machine = machine
    
    if best_machine is None:
        # Fallback: first machine
        best_machine = list(machine_options.keys())[0]
    
    return best_machine, machine_options[best_machine]


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Select the job with Most Work Remaining (MWR) - prioritize jobs with most total remaining work
    2. For that job's ready operation, select the best machine using SPT + load balancing
    
    This combines two effective heuristics:
    - MWR: Focus on jobs that have the most work left, reducing the chance of idle time
    - SPT with load balancing: Assign operations to machines that minimize completion time
    """
    if not ready_operations:
        return 0
    
    # Time guard: if we're running low on time, use a simple fallback
    if _time_left() < 0.5:
        # Simple MWR fallback
        max_remaining = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
                best_idx = i
        return best_idx
    
    # Compute machine loads from the ready operations' context
    # We don't have direct access to machine loads, so we estimate from the data
    # Each op has machine_options with durations. We can track approximate loads.
    
    # For simplicity and speed, use a heuristic that doesn't require full state:
    # 1. Pick job with most remaining work (MWR)
    # 2. Among that job's ready ops (there should be at most one ready op per job at a time),
    #    pick the machine with shortest duration (SPT)
    
    # First, group by job and find the job with max remaining work
    job_remaining = {}
    for op in ready_operations:
        job_id = op["job"]
        rw = op.get("remaining_work", 0)
        if job_id not in job_remaining or rw > job_remaining[job_id]:
            job_remaining[job_id] = rw
    
    # Find the job with maximum remaining work
    if not job_remaining:
        return 0
    
    max_job_remaining = max(job_remaining.values())
    
    # Find the ready operation for the job(s) with max remaining work
    # If multiple jobs have the same max, pick the one whose ready op has shortest min duration (SPT tie-break)
    candidates = []
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        rw = op.get("remaining_work", 0)
        if rw == max_job_remaining:
            # Get the minimum duration across machine options (SPT indicator)
            min_dur = min(op["machine_options"].values())
            candidates.append((i, min_dur))
    
    if not candidates:
        # Fallback: just pick the first one
        return 0
    
    # Among candidates with max remaining work, pick the one with shortest min duration (SPT)
    candidates.sort(key=lambda x: x[1])
    best_idx = candidates[0][0]
    
    return best_idx
# EVOLVE-BLOCK-END
