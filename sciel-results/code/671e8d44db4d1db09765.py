"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Explicit deadline guard: 150s wall-clock limit, use 145s for safety
_START_TIME = _time.monotonic()
_DEADLINE = _START_TIME + 145.0

# Cache for machine load tracking (since dispatch is called repeatedly)
# We track estimated machine availability to avoid scheduling on busy machines
_machine_load_cache = {}
_last_time_seen = -1


def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling using a combined approach:
    - Structural idea: Urgency/Cost ratio with critical-path estimation (from inspiration 2)
    - Refinement: Machine load tracking and soft-min cost (from inspiration 1 & 2)
    
    The key structural change: we track machine loads across dispatch calls to
    estimate which machines will be free sooner, and we use a ratio-based scoring
    that balances job urgency against expected wait time on the chosen machine.
    """
    global _machine_load_cache, _last_time_seen
    
    if _deadline_exceeded():
        # Fallback: pick max remaining work
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), 
                       key=lambda i: ready_operations[i]["remaining_work"]))
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Update time reference
    if time != _last_time_seen:
        _last_time_seen = time
        # Decay old load estimates (simple heuristic: loads are time-based)
        # We keep the cache but it's implicitly updated as new ops complete
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    best_machine_duration = float('inf')
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # No machine options; just score by remaining work
            score = remaining_work
            if score > best_score:
                best_score = score
                best_idx = idx
                best_machine = None
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        min_duration = min(durations)
        max_duration = max(durations)
        avg_duration = sum(durations) / len(durations)
        
        # --- Urgency calculation (critical-path inspired) ---
        # Estimate the critical path through remaining operations
        if remaining_ops > 1:
            # Smooth estimate: current op's expected duration + average remaining work per future op
            # Use a soft-max of durations to get a representative current duration
            if len(durations) > 1:
                temp = 0.15
                # Soft-max: approximates max but is smooth
                scaled = [d * temp for d in durations]
                max_scaled = max(scaled)
                log_sum_exp = max_scaled + math.log(sum(math.exp(s - max_scaled) for s in scaled))
                soft_max_duration = log_sum_exp / temp
            else:
                soft_max_duration = durations[0]
            
            # Remaining work after current operation
            remaining_after_current = max(0, remaining_work - soft_max_duration)
            # Average work per remaining operation
            avg_future_work = remaining_after_current / max(1, remaining_ops - 1)
            # Critical path estimate: current duration + estimated future path
            critical_path = soft_max_duration + avg_future_work
        else:
            critical_path = min_duration
        
        # Urgency: combine remaining work, critical path, and a penalty for many remaining ops
        # Jobs with high remaining work AND long critical paths are most urgent
        urgency = remaining_work + 0.4 * critical_path + 0.15 * max(0, 8 - remaining_ops)
        
        # --- Cost calculation (machine load aware) ---
        # For each machine, estimate when it would be available
        # We use the machine's current estimated load (from previous dispatches)
        # plus the processing time for this operation
        machine_costs = []
        for m in machines:
            d = machine_options[m]
            # Estimated load on this machine from previous assignments
            est_load = _machine_load_cache.get(m, 0)
            # Effective start time if we assign to this machine
            effective_cost = est_load + d
            machine_costs.append(effective_cost)
        
        # Soft-min over machine costs for smoothness
        if len(machine_costs) > 1:
            temp = 0.2
            scaled = [-c * temp for c in machine_costs]
            max_scaled = max(scaled)
            log_sum_exp = max_scaled + math.log(sum(math.exp(s - max_scaled) for s in scaled))
            soft_min_cost = -(log_sum_exp / temp)
        else:
            soft_min_cost = machine_costs[0] if machine_costs else min_duration
        
        cost = soft_min_cost + 1e-9
        
        # --- Score: urgency / cost ---
        score = urgency / cost
        
        if score > best_score:
            best_score = score
            best_idx = idx
            # Choose the machine that minimizes effective cost
            min_cost_idx = min(range(len(machine_costs)), key=lambda i: machine_costs[i])
            best_machine = machines[min_cost_idx]
            best_machine_duration = machine_options[best_machine]
    
    # Update machine load cache for the chosen assignment
    if best_machine is not None:
        current_load = _machine_load_cache.get(best_machine, 0)
        # The machine will be busy for at least the duration of this operation
        # starting from max(current time, current load)
        _machine_load_cache[best_machine] = max(time, current_load) + best_machine_duration
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
