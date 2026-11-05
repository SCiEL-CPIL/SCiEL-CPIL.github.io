"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a composite heuristic:
    - Primary: remaining work (critical path priority)
    - Secondary: prefer shorter processing times to reduce machine blocking
    - Tertiary: slight preference for operations with fewer machine options (constrained first)
    """
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the minimum processing time across available machines
        min_duration = min(machine_options.values()) if machine_options else 1
        
        # Number of machine options (fewer = more constrained)
        num_options = len(machine_options)
        
        # Composite score:
        # - High remaining work is good (priority to critical path)
        # - Shorter min duration is good (frees machine sooner)
        # - Fewer machine options is slightly better (constrained operations first)
        # 
        # Weight: remaining work dominates, but we normalize to avoid scale issues
        score = remaining_work * 1.0
        score -= min_duration * 0.1  # Small penalty for long processing
        score += (5 - num_options) * 0.05  # Small bonus for constrained ops
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
