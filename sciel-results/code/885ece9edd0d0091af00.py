"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Heuristic:
    - Primary: Maximize remaining_work (Longest Remaining Processing Time).
      This helps minimize makespan by ensuring long jobs are started early.
    - Secondary: Minimize min_duration (Shortest Processing Time on best machine).
      This breaks ties by preferring operations that can be completed quickly,
      potentially freeing up resources sooner or reducing blocking.
    - Machine Selection: Select the machine that offers the shortest processing
      time for the chosen operation.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    best_idx = -1
    best_score = float('-inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        rw = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue

        # Find the machine with the shortest duration for this operation
        # This is our "best machine" choice for this op
        best_machine_for_op = min(machine_options.keys(), key=lambda m: machine_options[m])
        min_dur = machine_options[best_machine_for_op]
        
        # Composite Score:
        # We want to prioritize high remaining work.
        # We use a large multiplier for remaining_work to make it the dominant factor.
        # We subtract a small penalty for duration to break ties in favor of shorter ops.
        score = rw * 10.0 - min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op

    # If no valid operation was found (should not happen if ready_operations is not empty)
    if best_idx == -1:
        return 0

    # Return (index, machine) to explicitly select the best machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
