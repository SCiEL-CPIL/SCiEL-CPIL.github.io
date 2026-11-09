"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

# Module-level deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Dynamic bottleneck optimization.
    Prioritize the operation that maximizes a score balancing:
    1. Job urgency: remaining_work (critical path proxy)
    2. Machine congestion avoidance: prefer machines with less estimated load
    3. Operation efficiency: prefer shorter processing times when possible
    
    Score = remaining_work / (1 + min_duration) * (1 / (1 + machine_load_proxy))
    
    The machine_load_proxy is approximated by the minimum duration across
    eligible machines (shorter = less congestion impact).
    
    Returns index of the selected ready operation.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Precompute scores for each ready operation
    # We want to maximize: urgency / congestion_penalty
    # urgency = remaining_work
    # congestion_penalty relates to how "expensive" this op is on its best machine
    
    best_idx = 0
    best_score = -1.0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        # Find minimum duration among eligible machines (best case)
        min_dur = min(machine_options.values())
        
        # Avoid division by zero
        if min_dur <= 0:
            min_dur = 1e-9
        
        # The ratio: more remaining work is better, but longer min duration is worse
        # This captures: "this job has a lot of work left, but this particular op
        # can be done quickly on some machine"
        ratio = remaining_work / min_dur
        
        # Additional tiebreaker: prefer operations with more remaining_ops in their job
        # (longer critical path ahead means more urgency)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Combined score: primary is ratio, secondary is remaining_ops
        score = ratio + 0.1 * remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
