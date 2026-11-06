"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = _time.monotonic() + 145.0


def _time_left():
    return _DEADLINE - _time.monotonic()


def _estimate_makespan_impact(ready_operations, time, chosen_idx):
    """
    Estimate the makespan impact of starting operation chosen_idx now.
    
    We simulate a short horizon: after starting the chosen op, we greedily
    schedule the remaining ready operations using a simple rule, and estimate
    the resulting completion time.
    
    Returns a lower value = better (shorter estimated makespan contribution).
    """
    n = len(ready_operations)
    if n <= 1:
        return 0.0

    # Build a lightweight model of the remaining work
    # For each job, track remaining work
    job_remaining = {}
    for op in ready_operations:
        j = op["job"]
        job_remaining[j] = job_remaining.get(j, 0) + op["remaining_work"]
    
    # If we start chosen_idx, that job's remaining work decreases by its min duration
    chosen_op = ready_operations[chosen_idx]
    chosen_job = chosen_op["job"]
    chosen_min_dur = min(chosen_op["machine_options"].values())
    job_remaining[chosen_job] -= chosen_min_dur
    
    # Simulate: after starting chosen op at 'time', it finishes at time + chosen_min_dur
    # The other ready ops must wait for their jobs or machines to be free.
    # Simple greedy: schedule remaining ops in order of (remaining_work desc, min_dur asc)
    
    # Estimate the critical path length from the remaining work
    # The makespan is at least max over all jobs of their remaining work,
    # but also constrained by machine availability.
    
    # For a quick estimate: the total remaining work across all jobs,
    # spread over the number of machines, gives a lower bound.
    total_remaining = sum(job_remaining.values())
    num_machines = max(len(op["machine_options"]) for op in ready_operations)
    lb_by_machines = total_remaining / max(num_machines, 1)
    
    # The job with the most remaining work sets a lower bound
    lb_by_job = max(job_remaining.values()) if job_remaining else 0.0
    
    # Combine: the makespan is at least max(lb_by_machines, lb_by_job)
    # Add the time already elapsed
    estimated_completion = time + max(lb_by_machines, lb_by_job)
    
    # Penalize if the chosen operation's job still has a lot of work
    # (meaning we didn't clear the critical path effectively)
    penalty = job_remaining.get(chosen_job, 0.0) * 0.5
    
    return estimated_completion + penalty


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Lookahead-based dispatch.
    For each candidate ready operation, estimate the makespan impact by
    simulating a short-term schedule. Choose the operation that leads to
    the lowest estimated completion time.
    
    This is a structural change from static scoring: we now consider the
    *consequences* of each choice rather than just intrinsic properties.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Safety: if we're running out of time, fall back to simple heuristic
    if _time_left() < 0.5:
        # Fallback: pick the operation with highest remaining_work
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_rw:
                best_rw = op["remaining_work"]
                best_idx = i
        return best_idx

    # For each candidate, estimate the makespan impact
    best_idx = 0
    best_score = float('inf')
    
    for i in range(n):
        score = _estimate_makespan_impact(ready_operations, time, i)
        if score < best_score:
            best_score = score
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
