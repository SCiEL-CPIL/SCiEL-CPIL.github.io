"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0

def _deadline_exceeded():
    return _time.time() >= _DEADLINE

def _soft_min(values, temperature=0.3):
    """Smooth surrogate for min using log-sum-exp (negative log-sum-exp for min)."""
    if not values:
        return 0.0
    min_val = min(values)
    if min_val <= 0:
        min_val = 1e-9
    shifted = [v - min_val for v in values]
    log_sum = math.log(sum(math.exp(v / temperature) for v in shifted))
    return min_val + temperature * log_sum

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy:
    - Priority: Maximize (remaining_work / soft_min_duration + 0.1 * remaining_ops)
      Uses soft-min (log-sum-exp) for smooth machine cost estimation.
    - Congestion penalty: Penalize operations whose best machine has high relative load.
    - Machine Selection: Select machine minimizing (duration + congestion_penalty).
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    # Pre-compute: for each machine, estimate its "load" from ready operations
    # A machine with many ready ops assigned to it is congested
    machine_load = {}
    for idx, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        for m in machine_options:
            machine_load[m] = machine_load.get(m, 0) + 1
    
    num_machines = len(machine_load) if machine_load else 1
    avg_load = sum(machine_load.values()) / num_machines if num_machines > 0 else 1
    
    W = 1.2  # Congestion penalty weight
    T = 0.3  # Soft-min temperature
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        # Use soft-min for smooth cost estimation
        sm = _soft_min(durations, temperature=T)
        if sm <= 0:
            sm = 1e-9
        
        # Core score: Urgency / Cost
        score = remaining_work / sm
        
        # Secondary: Prefer more remaining ops
        score += 0.1 * remaining_ops
        
        # Congestion bonus: favor ops whose best machine has low relative load
        # Find the best machine (min duration)
        best_m = min(machine_options.items(), key=lambda x: x[1])[0]
        best_m_load = machine_load.get(best_m, 0)
        rel_load = best_m_load / avg_load if avg_load > 0 else 1.0
        # Lower relative load -> higher bonus
        score += W * (1.0 - rel_load)
        
        if score > best_score:
            best_score = score
            best_idx = idx
            
            # Machine selection: minimize (duration + congestion_penalty * load)
            def machine_cost(item):
                m, d = item
                m_load = machine_load.get(m, 0)
                return d + W * 0.5 * (m_load / avg_load if avg_load > 0 else 1.0)
            
            best_machine = min(machine_options.items(), key=machine_cost)[0]
            
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
