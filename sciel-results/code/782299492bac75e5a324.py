"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0 # Or handle empty case appropriately, but usually not empty

    best_idx = -1
    best_score = -np.inf
    best_machine = None
    
    # Pre-compute min duration for each op to use as tie-breaker and for machine selection
    # We want to maximize remaining_work, and minimize completion time (time + duration)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        # Find the machine with the minimum duration for this operation
        min_duration = np.inf
        best_m = None
        for m, dur in machine_options.items():
            if dur < min_duration:
                min_duration = dur
                best_m = m
                
        if best_m is None:
            continue

        # Score: Primary is remaining_work (higher is better).
        # Secondary is earliest completion time (lower is better).
        # We can combine them. A common effective heuristic is simply max remaining_work.
        # Let's use a score that heavily favors remaining_work but breaks ties with speed.
        
        # To make it more robust, we can use a weighted sum if scales are comparable, 
        # but remaining_work is sum of durations, while time is absolute.
        # Let's stick to: Max remaining_work. Tie-break: Min duration.
        
        current_score = remaining_work
        # Add a small penalty for long processing time to break ties effectively
        # and slightly favor faster ops if remaining_work is identical.
        # The penalty should be small enough not to override remaining_work differences.
        # However, if remaining_work is the same, we definitely want the faster one.
        
        if current_score > best_score:
            best_score = current_score
            best_idx = i
            best_machine = best_m
        elif current_score == best_score:
            # Tie-breaker: prefer shorter duration (earlier completion)
            if min_duration < (np.inf if best_machine is None else 0): # Need to store best_min_duration
                pass # Logic below is cleaner

    # Let's re-implement the loop cleanly to store necessary tie-breaker info
    best_idx = -1
    best_remaining_work = -1
    best_min_duration = np.inf
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        min_duration = np.inf
        best_m = None
        for m, dur in machine_options.items():
            if dur < min_duration:
                min_duration = dur
                best_m = m
                
        if best_m is None:
            continue

        # 1. Higher remaining work is better
        # 2. If remaining work is equal, lower minimum duration is better
        if remaining_work > best_remaining_work:
            best_remaining_work = remaining_work
            best_min_duration = min_duration
            best_idx = i
            best_machine = best_m
        elif remaining_work == best_remaining_work:
            if min_duration < best_min_duration:
                best_min_duration = min_duration
                best_idx = i
                best_machine = best_m

    if best_idx == -1:
        return 0 # Fallback

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
