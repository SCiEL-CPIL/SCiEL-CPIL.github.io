"""Improved dispatch rule for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Module-level state for lightweight machine-load tracking (approximation)
# Since dispatch is called frequently, we keep a small cache keyed by (time, machine)
# but to keep it simple and correct, we'll compute a heuristic score per candidate.

def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    Returns an index into ready_operations.
    
    Heuristic: combine
      - remaining_work (higher is worse for makespan if left, so we prefer high remaining_work? 
        Actually, in scheduling, we often want to start operations that are critical.
        Let's think: if an operation has a lot of remaining work in its job, it might be critical.
        But also, we want to assign to the machine that will finish it earliest (least loaded).
    """
    if not ready_operations:
        return 0  # Should not happen, but safe

    # Compute a score for each ready operation.
    # We want to minimize makespan. Common heuristics:
    # 1. Longest Processing Time (LPT): pick the operation with the longest processing time.
    # 2. Most Remaining Work: pick the job with the most remaining work.
    # 3. Earliest Due Date: not applicable here.
    # 4. For flexible job-shop: pick the operation that, if started now on its best machine,
    #    leads to the earliest completion, considering machine loads.

    # Since we don't have machine load info directly, we can approximate:
    # For each operation, compute the minimum duration across its machine options.
    # Prefer operations with higher "urgency": e.g., high remaining_work / low min_duration?
    # Actually, a good heuristic is to pick the operation that has the highest remaining_work
    # among those with the shortest processing times? No, that's contradictory.

    # Let's use a combined score:
    # score = remaining_work / (min_duration + epsilon)
    # Higher score = more urgent.
    # But we also want to consider the machine choice. If we return (index, machine), we can pick the best machine.
    # The task says: "optionally (index, machine)". Let's try returning (index, machine) to allow better scheduling.

    best_score = -1
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue
        min_duration = min(machine_options.values())
        max_duration = max(machine_options.values())
        avg_duration = sum(machine_options.values()) / len(machine_options)
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Heuristic: prioritize operations with high remaining_work and short processing times.
        # Also, consider the number of remaining operations: if a job has few remaining ops,
        # finishing them quickly might reduce makespan.
        
        # Score: higher is better
        # Component 1: remaining_work (higher is more important)
        # Component 2: 1 / min_duration (shorter is better)
        # Component 3: 1 / remaining_ops (fewer remaining ops is more urgent)
        
        score = 0.0
        score += remaining_work * 1.0
        score += (1.0 / (min_duration + 1e-9)) * 5.0
        score += (1.0 / (remaining_ops + 1e-9)) * 2.0

        # Prefer machines with shorter durations for the chosen operation
        # Find the machine with the minimum duration
        best_m_for_op = min(machine_options, key=machine_options.get)

        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    # Return (index, machine) to allow the simulator to assign the best machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
