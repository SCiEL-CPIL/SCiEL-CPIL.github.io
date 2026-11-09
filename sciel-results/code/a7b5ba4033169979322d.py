"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation that minimizes the makespan contribution.
    
    Strategy:
    1. For each ready operation, identify the fastest eligible machine (min duration).
    2. Calculate a composite score that balances:
       - Critical path priority: operations with more remaining work/ops are more critical.
       - Shortest Processing Time (SPT): shorter operations reduce waiting times.
       - Load balancing: prefer machines that are likely less loaded (approximated by choosing
         the machine with the shortest duration, which implicitly tends to balance if we
         assume similar future loads, but we can also add a tie-breaker).
    
    We use a weighted score:
    score = alpha * (remaining_work / remaining_ops) + beta * (1 / min_duration)
    
    To minimize makespan, we often want to process operations that are on the critical path
    first (high remaining work) but also avoid long operations blocking the system (SPT).
    A common heuristic is to prioritize operations with high remaining work but short
    processing times on their best machine.
    
    Let's define:
    - criticality = remaining_work / max(1, remaining_ops)  # Average remaining work per op
    - urgency = 1 / (min_duration + epsilon)               # Inverse of processing time
    
    We want to maximize: criticality * urgency? Or a sum?
    Let's try maximizing: (remaining_work / remaining_ops) / min_duration
    This is similar to the current 'efficiency' but normalized by number of ops.
    
    Actually, the current approach is remaining_work / min_duration.
    Let's try a different angle: Longest Remaining Processing Time (LRPT) or Shortest?
    Usually, for makespan, Shortest Processing Time (SPT) is good for flowshop, but for
    job-shop with precedence, we might want to prioritize jobs that have many operations
    left (to keep them moving).
    
    Let's use a score that favors:
    1. High remaining work (critical path)
    2. Low processing time on the best machine (SPT)
    
    score = remaining_work / (min_duration * (1 + remaining_ops))
    This penalizes operations with many remaining steps if the duration is high?
    
    Alternative: Use the concept of "slack". But we don't have due dates.
    
    Let's stick to a robust heuristic:
    Priority = (remaining_work / remaining_ops) / min_duration
    This normalizes the total remaining work by the number of operations left, giving the
    average remaining work per operation, and then divides by the current processing time.
    High average work per op and short current time -> high priority.
    """
    if not ready_operations:
        return 0

    best_index = 0
    best_score = -float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        # Find the machine with the minimum duration for this operation
        best_m, min_duration = min(machine_options.items(), key=lambda item: item[1])
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        if remaining_ops <= 0:
            continue
            
        # Average remaining work per operation
        avg_remaining_work = remaining_work / remaining_ops
        
        # Score: prioritize high average remaining work and short processing time
        # This combines critical path (high work) with SPT (short time)
        score = avg_remaining_work / (min_duration + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

    if best_machine is None:
        return 0
        
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
