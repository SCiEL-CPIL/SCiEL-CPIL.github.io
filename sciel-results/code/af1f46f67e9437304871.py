"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math


def dispatch(ready_operations, time):
    """
    Dispatch rule using Longest Remaining Processing Time (LRPT).
    
    Heuristic:
    Priority = remaining_work (sum of min processing times for remaining ops)
    Tie-breaker = min_duration (shorter is better)
    
    Rationale:
    - Focuses on jobs with the most remaining work, assuming they are on the critical path.
    - This is a "Longest Job First" strategy, distinct from the "Weighted Shortest Job First" 
      approach of the seed.
    - By starting the longest jobs early, we minimize the risk of them becoming the bottleneck
      at the end of the schedule.
    """
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_score = float('-inf')
    best_tie_breaker = float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        if not machine_options:
            continue
            
        # Primary score: Remaining work (higher is better)
        # If remaining_work is not present or 0, we might need a fallback.
        # Let's assume it's present.
        primary_score = op.get("remaining_work", 0)
        
        # Secondary score: Min duration (lower is better)
        min_duration = min(machine_options.values())
        
        # Compare
        # We want to maximize primary_score.
        # If primary_score is equal, we want to minimize min_duration.
        
        if primary_score > best_score:
            best_score = primary_score
            best_tie_breaker = min_duration
            best_idx = i
            best_machine = min(machine_options, key=machine_options.get)
        elif primary_score == best_score:
            if min_duration < best_tie_breaker:
                best_tie_breaker = min_duration
                best_idx = i
                best_machine = min(machine_options, key=machine_options.get)
                
    # Fallback if no valid scores (e.g., remaining_work is 0 for all)
    if best_score <= 0:
        best_idx = 0
        best_tie_breaker = float('inf')
        for i, op in enumerate(ready_operations):
            if not op["machine_options"]:
                continue
            min_duration = min(op["machine_options"].values())
            if min_duration < best_tie_breaker:
                best_tie_breaker = min_duration
                best_idx = i
                best_machine = min(op["machine_options"], key=op["machine_options"].get)

    return (int(best_idx), best_machine)
# EVOLVE-BLOCK-END
