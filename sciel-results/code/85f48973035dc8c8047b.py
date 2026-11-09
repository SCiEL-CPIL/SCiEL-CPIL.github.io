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

def _soft_min(values, temperature=1.0):
    """Soft-min via log-sum-exp for smooth approximation of min."""
    if not values:
        return 0.0
    min_v = min(values)
    if temperature <= 0:
        return min_v
    # Soft-min = -log(sum(exp(-v/T)))/T
    scaled = [-v / temperature for v in values]
    max_scaled = max(scaled)
    log_sum = math.log(sum(math.exp(s - max_scaled) for s in scaled)) + max_scaled
    return -log_sum * temperature

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
    
    # Precompute normalization constants
    max_rw = max(op.get("remaining_work", 0) for op in ready_operations)
    if max_rw <= 0:
        max_rw = 1.0
    
    # Estimate machine loads by summing durations across ready ops
    # This helps identify bottleneck machines
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            machine_load[m] = machine_load.get(m, 0) + d
    
    # Average load per machine
    if machine_load:
        avg_load = sum(machine_load.values()) / len(machine_load)
    else:
        avg_load = 1.0
    
    # Weights: critical path (remaining work) is dominant
    w_rw = 0.55
    w_spt = 0.25
    w_load = 0.20
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 1)
        
        if not mo:
            min_d = 1.0
            best_machine_load = avg_load
        else:
            durations = list(mo.values())
            min_d = min(durations)
            
            # Use soft-min to consider flexibility: if multiple machines have similar durations,
            # the effective cost is lower due to flexibility
            soft_min_d = _soft_min(durations, temperature=0.5)
            
            # Find the machine with minimum load among those with near-minimal duration
            # This balances load while respecting efficiency
            min_d_threshold = min_d * 1.2  # Allow up to 20% longer if it reduces load
            candidate_machines = [(m, d) for m, d in mo.items() if d <= min_d_threshold]
            
            if candidate_machines:
                best_machine_load = min(machine_load.get(m, 0) for m, d in candidate_machines)
            else:
                best_machine_load = machine_load.get(list(mo.keys())[0], 0)
        
        # Normalize components
        norm_rw = rw / max_rw
        
        # SPT component: shorter processing time is better
        # Use inverse with offset to avoid division by zero
        norm_spt = 1.0 / (min_d + 1.0)
        
        # Load balancing: prefer operations that can run on less-loaded machines
        # Lower load is better, so invert
        if avg_load > 0:
            norm_load = 1.0 - (best_machine_load / (2.0 * avg_load))
            norm_load = max(0.0, min(1.0, norm_load))
        else:
            norm_load = 0.5
        
        # Critical path bonus: operations with fewer remaining steps but high remaining work
        # are on the critical path and should be prioritized
        # This captures the idea that we want to unblock jobs that are close to completion
        # but still have significant work
        if remaining_ops > 0:
            critical_bonus = 1.0 / (1.0 + remaining_ops)
        else:
            critical_bonus = 0.0
        
        # Combined score
        score = (w_rw * norm_rw + 
                 w_spt * norm_spt + 
                 w_load * norm_load +
                 0.05 * critical_bonus)
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer shorter processing time, then lower load
            prev_op = ready_operations[best_idx]
            prev_mo = prev_op.get("machine_options", {})
            prev_min_d = min(prev_mo.values()) if prev_mo else 1.0
            if min_d < prev_min_d:
                best_idx = i
            elif abs(min_d - prev_min_d) < 1e-9:
                prev_best_load = machine_load.get(list(prev_mo.keys())[0], 0) if prev_mo else 0
                if best_machine_load < prev_best_load:
                    best_idx = i
    
    return best_idx

# EVOLVE-BLOCK-END
