"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard: evaluator kills after 150s wall-clock; we want to return well before.
# We track a global start time and a per-call budget.
_start_time = None
_CALL_BUDGET = 0.005  # seconds per call max (5ms) - very safe for 10 jobs * 5 machines

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.perf_counter()
    return _time.perf_counter() - _start_time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: For each ready operation, compute a score based on:
    1. The maximum "urgency" across its machine options: 
       urgency = remaining_work / min_duration (prioritize jobs with lots of work 
       that can be done quickly on their best machine)
    2. Tie-break: prefer the operation whose best machine has the least remaining 
       total work assigned (to balance machine loads).
    
    This approximates a critical-path / longest-processing-time hybrid that 
    tends to reduce makespan in open-shop settings.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Track elapsed to ensure we stay within budget
    elapsed = _get_elapsed()
    if elapsed > 145.0:
        # Emergency: just return 0
        return 0
    
    # Pre-compute machine load estimates from remaining_work of all ready ops
    # For each machine, sum up the minimum duration contribution from each ready op
    # that can use that machine. This gives a proxy for machine congestion.
    machine_load = {}
    for op in ready_operations:
        opts = op["machine_options"]
        for m, dur in opts.items():
            if m not in machine_load:
                machine_load[m] = 0.0
            # Use the duration on this machine as a proxy for load contribution
            machine_load[m] += dur
    
    # Score each operation
    best_idx = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            continue
        
        # Find the best machine for this operation (shortest duration)
        min_dur = min(opts.values())
        best_machine = min(opts, key=lambda m: opts[m])
        
        # Primary score: remaining work normalized by the best possible duration
        # This favors operations that have a lot of work but can be done quickly
        urgency = op["remaining_work"] / max(min_dur, 1e-9)
        
        # Secondary: prefer machines with lower estimated load (congestion avoidance)
        load_penalty = machine_load.get(best_machine, 0.0)
        
        # Combine: high urgency, low congestion
        score = urgency - 0.1 * load_penalty
        
        # Tertiary: slightly prefer operations with fewer remaining ops (finishing jobs)
        remaining_ops_penalty = 0.05 * op["remaining_ops"]
        score -= remaining_ops_penalty
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
