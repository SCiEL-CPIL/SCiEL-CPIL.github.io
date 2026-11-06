"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Cache for machine load tracking across dispatch calls
_machine_load_cache = {}


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Use the current time as a key to track machine loads
    # In an event-driven simulation, 'time' changes as operations complete
    # We use a simple heuristic: prioritize operations that reduce makespan
    
    best_idx = 0
    best_score = float('-inf')
    
    n = len(ready_operations)
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # No machine options, use a default score
            score = op.get("remaining_work", 0) * 10 + op.get("remaining_ops", 0) * 5
        else:
            # Find the machine with minimum duration (preferred machine)
            min_duration = min(machine_options.values())
            preferred_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            
            # Score components:
            # 1. Prefer operations on less-loaded machines (we approximate load by time)
            # 2. Prefer jobs with more remaining work (critical path)
            # 3. Prefer shorter processing times (reduces makespan)
            
            remaining_work = op.get("remaining_work", 0)
            remaining_ops = op.get("remaining_ops", 0)
            
            # Heuristic: weight by remaining work and ops, penalize long durations
            # Use a combination that balances job urgency and machine utilization
            score = (
                remaining_work * 2.0 +          # Job has more total work left
                remaining_ops * 1.0 +            # Job has more operations left
                (1.0 / (min_duration + 1e-9)) * 5.0  # Prefer shorter processing times
            )
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
