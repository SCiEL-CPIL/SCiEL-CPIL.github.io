"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import defaultdict

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill

# Global state to track estimated machine loads
# We maintain a running estimate of when each machine will be free
_machine_free_time = defaultdict(float)
_last_dispatch_time = 0.0


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Structural change: Model dispatch as dynamic bottleneck mitigation.
    - Estimate the completion time for each (operation, machine) pair.
    - Prioritize operations that minimize the estimated completion time relative
      to the current global bottleneck (max machine free time).
    - This routes work to least congested resources, reducing queue buildup.
    
    Returns: (index, machine) tuple to explicitly select the best machine.
    """
    global _machine_free_time, _last_dispatch_time
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    # Update machine free times based on elapsed time
    # This is a heuristic update: we assume machines have been busy since last dispatch
    elapsed = time - _last_dispatch_time
    _last_dispatch_time = time

    # Find the current global bottleneck (max machine free time)
    if _machine_free_time:
        global_bottleneck = max(_machine_free_time.values())
    else:
        global_bottleneck = time

    best_idx = -1
    best_score = float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        rw = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue

        # For each eligible machine, estimate completion time
        for machine, duration in machine_options.items():
            # Estimated free time for this machine
            machine_free = _machine_free_time.get(machine, time)
            
            # The operation can start at max(current_time, machine_free)
            # But since we're dispatching now, the operation starts at current time
            # if the machine is free, or waits if not.
            # In an event-driven simulation, if the machine is busy, the op queues.
            # The estimated completion time is max(time, machine_free) + duration
            
            est_completion = max(time, machine_free) + duration
            
            # Score: We want to minimize the estimated completion time.
            # But we also want to prioritize jobs with more remaining work to avoid
            # starving long jobs. We use a composite score.
            
            # Normalize completion time relative to global bottleneck
            # Lower is better. We want to pick the op that minimizes this.
            # Add a penalty for high remaining work to prioritize long jobs?
            # No, we want to START long jobs early. So we should prefer ops
            # with high remaining work.
            
            # Let's use: score = est_completion - alpha * rw
            # This way, higher rw reduces the score (making it more attractive),
            # and lower est_completion also reduces the score.
            
            alpha = 0.1  # Weight for remaining work
            score = est_completion - alpha * rw
            
            if score < best_score:
                best_score = score
                best_idx = i
                best_machine = machine

    # If no valid operation was found
    if best_idx == -1:
        return 0

    # Update the machine free time for the selected machine
    # The machine will be busy until est_completion
    if best_machine is not None:
        machine_free = _machine_free_time.get(best_machine, time)
        est_completion = max(time, machine_free) + machine_options[best_machine]
        _machine_free_time[best_machine] = est_completion

    # Return (index, machine) to explicitly select the best machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
