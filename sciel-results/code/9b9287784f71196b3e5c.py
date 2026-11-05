"""Job-shop dispatch rule: Critical Path (LRPT) with Optimal Machine Assignment."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Primary: Maximize Remaining Work (LRPT / Critical Path Length).
       This ensures that jobs with the longest remaining paths are prioritized,
       directly targeting the reduction of the makespan.
    2. Secondary: Minimize Estimated Completion Time of the selected operation.
       For each operation, we evaluate all machine options and select the machine
       that minimizes a cost function combining duration and current contention
       (number of ready operations targeting that machine).
    3. We return the index of the best operation and the specific machine chosen
       for it, allowing the simulator to use our optimized assignment.
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        # Single op: pick best machine for it
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            # Find machine with min duration (assuming no contention info needed for single)
            # But we can still check counts if we want, though n=1 means count=1 for all
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Step 1: Count how many ready operations target each machine
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    best_idx = 0
    best_machine = None
    best_score = float("-inf")

    # Weights
    W_RW = 1.0
    W_COST = 0.5
    
    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        
        mo = op.get("machine_options", {})
        if not mo:
            # Fallback if no machine options provided
            score = W_RW * rw
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = None
            continue

        # Find the best machine for this operation
        # Cost = duration + contention_penalty
        # Contention penalty is proportional to the number of other ready ops wanting that machine
        min_cost = float("inf")
        chosen_m = None
        
        for m, d in mo.items():
            c = machine_counts.get(m, 1)
            # The "cost" represents the estimated time this op will occupy the machine
            # or the delay it incurs.
            # A simple linear model: time = duration + k * (count - 1)
            # If count is 1, no wait. If count is 2, some wait.
            # Let's use d * (1 + 0.5 * (c - 1)) as a smooth-ish estimate
            cost = d * (1.0 + 0.5 * (c - 1))
            if cost < min_cost:
                min_cost = cost
                chosen_m = m
        
        # Score: High RW is good, Low Cost is good
        score = W_RW * rw - W_COST * min_cost
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_m

    if best_machine is not None:
        return (int(best_idx), best_machine)
    else:
        return int(best_idx)

# EVOLVE-BLOCK-END
