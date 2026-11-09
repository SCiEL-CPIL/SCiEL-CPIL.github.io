"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np
from collections import defaultdict

# Module-level deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill

# Track machine loads to estimate congestion
_machine_loads = defaultdict(float)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Dynamic bottleneck optimization with congestion awareness.
    Prioritize the operation that maximizes a score balancing:
    1. Job urgency: remaining_work (critical path proxy)
    2. Machine congestion avoidance: prefer machines with less estimated load
    3. Operation efficiency: prefer shorter processing times when possible
    
    Uses soft-min to smoothly estimate the best machine duration while
    accounting for alternative machine options.
    
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
    # congestion_penalty relates to how "expensive" this op is considering machine loads
    
    best_idx = 0
    best_score = -1.0
    
    # Temperature for soft-min (lower = more like hard min)
    soft_min_temp = 0.5
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            continue
        
        # Compute soft-min of durations across eligible machines
        # This smoothly balances between the best machine and alternatives
        durations = list(machine_options.values())
        
        if len(durations) == 1:
            effective_dur = durations[0]
        else:
            # Soft-min using log-sum-exp trick: -log(sum(exp(-d/T)))
            # This gives a value close to min(d) but smoother
            neg_d_over_t = np.array([-d / soft_min_temp for d in durations])
            max_val = np.max(neg_d_over_t)
            soft_min_val = -soft_min_temp * (max_val + np.log(np.sum(np.exp(neg_d_over_t - max_val))))
            effective_dur = max(soft_min_val, 1e-9)
        
        # Estimate congestion: how loaded are the eligible machines?
        # Higher load means this operation will wait longer or cause more delay
        congestion_score = 0.0
        if machine_options:
            # Average load of eligible machines
            loads = [_machine_loads.get(m, 0.0) for m in machine_options.keys()]
            avg_load = np.mean(loads) if loads else 0.0
            congestion_score = avg_load
        
        # The ratio: more remaining work is better, but longer effective duration is worse
        # Higher congestion is worse
        if effective_dur <= 0:
            effective_dur = 1e-9
        
        # Primary score: urgency relative to processing time
        urgency_ratio = remaining_work / effective_dur
        
        # Congestion penalty: operations on heavily loaded machines are penalized
        congestion_penalty = 1.0 / (1.0 + congestion_score * 0.1)
        
        # Combined score
        score = urgency_ratio * congestion_penalty + 0.1 * remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    # Update machine load estimates based on the selected operation
    # This helps future dispatches account for current congestion
    selected_op = ready_operations[best_idx]
    machine_options = selected_op.get("machine_options", {})
    if machine_options:
        # Estimate which machine will be used (the one with best load-adjusted duration)
        best_machine = None
        best_adjusted_dur = float('inf')
        for m, dur in machine_options.items():
            load = _machine_loads.get(m, 0.0)
            adjusted = dur * (1.0 + load * 0.05)  # Penalize loaded machines
            if adjusted < best_adjusted_dur:
                best_adjusted_dur = adjusted
                best_machine = m
        
        if best_machine is not None:
            _machine_loads[best_machine] += best_adjusted_dur
    
    return int(best_idx)
# EVOLVE-BLOCK-END
