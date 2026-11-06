"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time_mod
import numpy as np

_DEADLINE = _time_mod.monotonic() + 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next in an open-shop simulation.
    
    Strategy:
    1. Select the job with the most remaining work (MWR) to reduce makespan.
    2. Among its ready operations, pick the one with the shortest duration on
       its assigned machine (SPT-like) to quickly clear short tasks.
    3. If machine selection is allowed, prefer the machine that minimizes the
       resulting load imbalance.
    """
    # Deadline guard
    if _time_mod.monotonic() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    # Step 1: Score each ready operation
    # We want to pick the operation belonging to the job with the most remaining work,
    # and among those, the one that finishes soonest (shortest duration).
    
    # Pre-compute job remaining work
    job_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        if j not in job_work:
            job_work[j] = op["remaining_work"]
    
    # For each operation, compute:
    #   - job_remaining_work (higher is better)
    #   - min_duration on its machine options (lower is better)
    #   - We combine: prioritize high job work, then short duration
    
    best_idx = 0
    best_key = None
    
    for i, op in enumerate(ready_operations):
        j = op["job"]
        work = op["remaining_work"]
        
        # Get the minimum duration among machine options
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict):
            durations = list(machine_options.values())
        else:
            durations = list(machine_options)
        
        if not durations:
            min_dur = float('inf')
        else:
            min_dur = min(durations)
        
        # Key: primarily by job remaining work (descending), secondarily by min duration (ascending)
        # We want to maximize work, minimize duration
        # Use a tuple: (work, -min_dur) so that max works
        key = (work, -min_dur)
        
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
