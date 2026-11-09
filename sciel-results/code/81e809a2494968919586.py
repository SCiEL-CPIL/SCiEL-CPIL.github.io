"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Dispatch rule using bipartite matching concept between ready operations and machines.
    Selects the operation-machine pair that maximizes the minimum remaining slack time
    across all jobs and machines, prioritizing operations most constrained by their
    latest possible start times to prevent future bottlenecks.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Build bipartite matching structure
    # For each ready operation, consider all (operation, machine) pairs
    # Score each pair based on resulting minimum slack across all jobs and machines
    
    # First, compute aggregate remaining work per job and per machine
    job_remaining = defaultdict(float)
    machine_remaining = defaultdict(float)
    
    for op in ready_operations:
        job = op["job"]
        # Use the minimum duration as the job's contribution (optimistic estimate)
        min_dur = min(op["machine_options"].values()) if op["machine_options"] else 0
        job_remaining[job] += min_dur
        
        for machine, duration in op["machine_options"].items():
            machine_remaining[machine] += duration
    
    # For each operation-machine pair, compute the minimum slack that would result
    # if we scheduled this operation on this machine
    # Slack for a job = remaining work on that job
    # Slack for a machine = remaining work on that machine
    # We want to maximize the minimum slack (most balanced schedule)
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        options = op["machine_options"]
        
        if not options:
            continue
        
        # Try each machine option for this operation
        for machine, duration in options.items():
            # Compute resulting state if we schedule op on machine
            # Job remaining after: job_remaining[job] - duration (this op's contribution)
            # Machine remaining after: machine_remaining[machine] - duration
            
            # Estimate the minimum slack across all jobs and machines
            # For simplicity, use the minimum of:
            # - All job remainings (excluding this job's reduction)
            # - All machine remainings (excluding this machine's reduction)
            
            # Compute min job slack
            min_job_slack = float('inf')
            for j, work in job_remaining.items():
                if j == job:
                    adjusted = work - duration
                else:
                    adjusted = work
                if adjusted < min_job_slack:
                    min_job_slack = adjusted
            
            # Compute min machine slack
            min_machine_slack = float('inf')
            for m, work in machine_remaining.items():
                if m == machine:
                    adjusted = work - duration
                else:
                    adjusted = work
                if adjusted < min_machine_slack:
                    min_machine_slack = adjusted
            
            # The bottleneck is the minimum of job and machine slacks
            min_slack = min(min_job_slack, min_machine_slack)
            
            # We want to MAXIMIZE this minimum slack (most balanced)
            if min_slack > best_score:
                best_score = min_slack
                best_idx = i
    
    # Fallback: if no valid pair found, use remaining_work as tiebreaker
    if best_score == -float('inf'):
        scores = [op["remaining_work"] for op in ready_operations]
        best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
