"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 10  # seconds before the 120s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Calculate a composite priority score for each ready operation.
    2. Primary signal: remaining_work (criticality of the job).
    3. Secondary signal: ratio of remaining_work to min_duration (efficiency).
    4. Tertiary signal: MPT preference (max duration) to commit machines to long blocks.
    5. For the selected operation, assign it to the machine with the minimum 
       duration to ensure it finishes as soon as possible.
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    scores = []
    best_machines = []
    
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = op["machine_options"]
        
        if not durations:
            scores.append(0)
            best_machines.append(None)
            continue
            
        durs = list(durations.values())
        min_dur = min(durs)
        max_dur = max(durs)
        
        # Avoid division by zero
        if min_dur <= 0:
            min_dur = 1e-9
            
        # Composite score:
        # - rw is the dominant factor (critical path)
        # - rw / min_dur emphasizes efficiency (high work, short time)
        # - max_dur adds MPT preference for tie-breaking
        # Weighting: rw dominates, then efficiency ratio, then MPT
        score = rw * 1e6 + (rw / min_dur) * 1e3 + max_dur
        
        scores.append(score)
        best_m, _ = _best_machine_for_op(op)
        best_machines.append(best_m)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    Returns (machine, duration).
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END
