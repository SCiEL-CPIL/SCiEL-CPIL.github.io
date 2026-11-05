"""Open-shop dispatch rule: machine-load balancing with constraint-aware scoring."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Machine-load balancing. For each ready operation, identify its
    most constrained machine (highest load proxy / fewest alternatives), and
    prioritize operations that would relieve the most loaded machines.
    This prevents any single machine from becoming the dominant bottleneck.
    
    Score per operation:
      - Primary: remaining_work on the most-constrained machine (higher = more urgent to schedule)
      - Secondary: total remaining_work (tiebreaker)
      - Tertiary: fewer remaining_ops (sooner completion)
    
    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0
    
    # Build machine load proxy: for each machine, sum of durations of ready ops
    # that can use it. This gives a proxy for how "loaded" each machine is.
    machine_load = {}
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0.0) + dur
    
    # Also count how many ready ops can use each machine (constraint proxy)
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    # For each operation, compute a score
    # The idea: an operation is "urgent" if it uses a machine that is heavily
    # loaded (high total remaining work on that machine) AND that machine has
    # few alternatives (low count means it's a bottleneck).
    
    scores = []
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            scores.append(-1.0)
            continue
        
        # Find the most constrained machine for this operation
        # Constrained = high load + few alternatives
        best_constraint = -1.0
        for m, dur in opts.items():
            load = machine_load.get(m, 0.0)
            count = machine_count.get(m, 1)
            # Constraint score: high load and low count = very constrained
            # Use load / count as a normalized pressure
            constraint = load / max(count, 1)
            # Weight by how much work this op puts on that machine relative to total
            # Actually, we want to prioritize ops that USE the most constrained machine
            # The duration on that machine matters: longer duration on a constrained machine = more urgent
            constraint = constraint * dur
            if constraint > best_constraint:
                best_constraint = constraint
        
        # Combine with total remaining work and remaining ops
        rem_work = op.get("remaining_work", sum(opts.values()))
        rem_ops = op.get("remaining_ops", len(opts))
        
        # Score: prioritize high constraint (bottleneck relief), then more remaining work
        # We want to pick the operation that, if started now, most reduces the
        # critical path. Starting an op that uses a heavily loaded machine with
        # few alternatives frees up that machine sooner.
        score = best_constraint + 0.1 * rem_work - 0.01 * rem_ops
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
