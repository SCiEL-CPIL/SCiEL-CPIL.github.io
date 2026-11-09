"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def _fast_dispatch(ready_operations, time):
    """Extremely fast fallback: Longest Remaining Work (LRW) / MWR."""
    if not ready_operations:
        return 0
    max_rw = -1
    best_idx = 0
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
            best_idx = i
    return best_idx

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        return _fast_dispatch(ready_operations, time)
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Estimate machine loads using weighted distribution across all machine options
    # Weight each machine by inverse of its duration (shorter = higher weight)
    machine_load = {}
    total_weight = 0.0
    
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        # Calculate weights: 1/duration for each machine
        weights = {}
        sum_weights = 0.0
        for m, d in mo.items():
            w = 1.0 / (d + 1e-9)  # Avoid division by zero
            weights[m] = w
            sum_weights += w
        
        # Distribute load proportionally to weights
        for m, w in weights.items():
            share = w / sum_weights
            # The contribution to load is the expected processing time on that machine
            # weighted by probability of assignment
            machine_load[m] = machine_load.get(m, 0.0) + share * (1.0 / sum_weights * sum(weights.values()))
    
    # Simpler approach: just sum up the minimum duration for each machine
    # and use soft-min for smoothing
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        # For each machine, add its duration to the load
        # This represents the total work that could be assigned to each machine
        for m, d in mo.items():
            machine_load[m] = machine_load.get(m, 0.0) + d
    
    if not machine_load:
        return _fast_dispatch(ready_operations, time)
    
    # Calculate average load for normalization
    avg_load = sum(machine_load.values()) / len(machine_load)
    if avg_load <= 0:
        avg_load = 1.0
    
    # SRPT-based scoring: prioritize operations with shortest remaining processing time
    # weighted by inverse of machine congestion
    best_score = float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        rw = op.get("remaining_work", 0)
        
        if not mo:
            # Fallback: use remaining work as proxy
            score = rw
        else:
            # Find the best machine for this operation
            # Score each machine option: duration * (1 + normalized_load)
            # Lower score = better (less congestion + shorter time)
            min_score = float('inf')
            
            for m, d in mo.items():
                load = machine_load.get(m, 0.0)
                # Normalize load: higher load = higher penalty
                # Use soft normalization to avoid extreme values
                norm_load = load / avg_load
                # Effective cost: duration adjusted by machine congestion
                # We want to minimize this
                cost = d * (1.0 + 0.5 * norm_load)
                if cost < min_score:
                    min_score = cost
            
            # Combine with urgency: operations with more remaining work are more urgent
            # But SRPT focuses on shortest processing time
            # Use a weighted combination:
            # - Primary: minimize effective cost (machine congestion + duration)
            # - Secondary: prioritize operations with more remaining work (urgency)
            
            # Normalize min_score to [0, 1] range
            # Use inverse: lower cost = higher priority
            if min_score > 0:
                norm_cost = 1.0 / (1.0 + min_score / avg_load)
            else:
                norm_cost = 1.0
            
            # Urgency component: higher remaining work = more urgent
            # But we want to balance this with SRPT
            # Use a soft urgency: log(remaining_work + 1) to avoid extreme values
            urgency = math.log(rw + 1.0) if rw > 0 else 0.0
            
            # Combined score: we want to minimize this
            # High urgency should be prioritized, but not at the expense of machine balance
            score = min_score * (1.0 - 0.1 * min(urgency / 10.0, 1.0))
        
        if score < best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer shorter processing time
            prev_mo = ready_operations[best_idx].get("machine_options", {})
            cur_mo = op.get("machine_options", {})
            
            prev_min_d = min(prev_mo.values()) if prev_mo else float('inf')
            cur_min_d = min(cur_mo.values()) if cur_mo else float('inf')
            
            if cur_min_d < prev_min_d:
                best_idx = i
    
    return best_idx

# EVOLVE-BLOCK-END
