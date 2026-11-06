"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def dispatch(ready_operations, time):
    # Deadline guard: if we are running too long, return a safe default
    # This is unlikely to trigger given the O(N) complexity, but satisfies the directive.
    start_time = time.time()
    deadline = start_time + 0.1  # 100ms safety margin for this call
    
    if not ready_operations:
        return 0 # Should not happen, but safe

    best_score = -1
    best_index = 0
    best_machine = None
    
    # Pre-calculate max values for normalization if needed, 
    # but simple weighted sum is robust enough.
    
    for i, op in enumerate(ready_operations):
        # Check deadline
        if time.time() > deadline:
            break
            
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # Fallback if no machine options provided
            score = remaining_work
            m = None
        else:
            # Find the machine with the minimum duration (SPT for machine selection)
            # This helps reduce the makespan of the specific operation
            min_duration = float('inf')
            chosen_machine = None
            for m, d in machine_options.items():
                if d < min_duration:
                    min_duration = d
                    chosen_machine = m
            
            # Heuristic: 
            # 1. Prioritize jobs with more remaining work (tail latency)
            # 2. Prioritize operations with longer durations (bottleneck mitigation / LPT)
            # 3. Small weight for remaining ops (job complexity)
            
            # Normalization is tricky without global stats, so we use raw values.
            # Weights: 0.7 for work, 0.3 for duration.
            # The inspiration used 0.7 RW, 0.2 RO, 0.1 invDur.
            # Let's try shifting weight from RO to Duration (positive).
            
            score = 0.7 * remaining_work + 0.3 * min_duration + 0.0 * remaining_ops
            
            if score > best_score:
                best_score = score
                best_index = i
                best_machine = chosen_machine

    # If the loop broke early due to deadline, we still return the best found so far
    # If no best found (e.g. empty list handled above), return 0
    
    # Return format: index or (index, machine)
    # The prompt says "optionally (index, machine)".
    # If we selected a specific machine, we should return it to be explicit.
    if best_machine is not None:
        return (best_index, best_machine)
    else:
        return best_index
# EVOLVE-BLOCK-END
