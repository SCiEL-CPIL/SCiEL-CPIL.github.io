"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s margin before 150s limit

def dispatch(ready_operations, time):
    # Check deadline
    if time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Extract duration
        # machine_options is {machine: duration}
        machine_options = op.get("machine_options", {})
        if machine_options:
            # Use min duration as a proxy for "efficiency" or "min time to complete"
            # If there's only one, it's that one.
            duration = min(machine_options.values())
        else:
            duration = 0

        # Heuristic: 
        # Primary: Remaining Work (LRPT)
        # Secondary: Remaining Ops (Complexity/Risk)
        # Tertiary: Current Duration (LPT for machine utilization)
        
        # Weights tuned based on "Measured: 1 -> 1.002" hint
        score = remaining_work + 0.1 * remaining_ops + 0.1 * duration
        scores.append(score)
    
    # Find index of max score
    # If ties, np.argmax picks the first one.
    # We can add a tiny random tie-breaker? No, keep deterministic.
    # However, if scores are identical, we might want to break ties by job ID or something?
    # The seed just used argmax.
    
    try:
        idx = int(np.argmax(scores))
    except:
        idx = 0
        
    return idx
# EVOLVE-BLOCK-END
