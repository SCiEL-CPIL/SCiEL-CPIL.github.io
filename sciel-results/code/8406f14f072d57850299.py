"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.monotonic() + 140.0  # safety margin before 150s kill

def dispatch(ready_operations, time):
    # Deadline guard: if we're running long, fall back to a fast heuristic
    if _time.monotonic() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # For each ready operation, compute:
    # 1. min_duration: the shortest processing time among its machine options
    # 2. max_duration: the longest processing time
    # 3. remaining_work: total work left in the job
    # 4. remaining_ops: number of ops left in the job
    #
    # Priority score combines:
    # - Workload urgency: jobs with more remaining work should be prioritized
    # - Efficiency: operations that can be done quickly (low min_duration) reduce
    #   contention and free up machines sooner
    # - Tightness: operations with few remaining ops but high work are critical path
    #
    # We use a composite score:
    #   score = remaining_work / (min_duration + epsilon) * (1 + 1/remaining_ops)
    #
    # This prioritizes:
    # - High workload (more work to clear)
    # - Low min duration (can be executed quickly, reducing makespan contribution)
    # - Few remaining ops (closer to completion, finishing them helps overall makespan)

    eps = 1e-9
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        durations = list(mo.values())
        min_dur = min(durations) if durations else 1.0
        rem_work = op.get("remaining_work", sum(durations))
        rem_ops = op.get("remaining_ops", 1)
        
        # Composite score: prioritize high work, low min duration, few remaining ops
        # Higher score = higher priority
        urgency = rem_work / (min_dur + eps)
        completion_bonus = 1.0 + 1.0 / max(rem_ops, 1)
        scores[i] = urgency * completion_bonus

    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))

    # If the operation has multiple machine options, we could return (index, machine)
    # but the interface says "optionally (index, machine)". The seed returns just index.
    # Let's check if returning machine helps. For now, return just the index as in seed,
    # but we could enhance by picking the best machine.
    # The task says: "dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))"
    # Returning just the index is safe and matches the seed interface.
    
    return best_idx
# EVOLVE-BLOCK-END
