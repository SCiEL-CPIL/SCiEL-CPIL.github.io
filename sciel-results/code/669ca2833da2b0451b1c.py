"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

# Track machine loads across dispatch calls
_machine_loads = {}

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

def _get_machine_load(machine):
    """Get current estimated load for a machine."""
    return _machine_loads.get(machine, 0.0)

def _update_machine_load(machine, duration):
    """Update the tracked load for a machine."""
    _machine_loads[machine] = _machine_loads.get(machine, 0.0) + duration

def _compute_greedy_assignment(ready_operations):
    """
    Simulate greedy SPT assignment to estimate machine loads.
    Returns a dict of machine -> estimated completion time if all ready ops were assigned.
    """
    # Start with current known loads
    sim_loads = dict(_machine_loads)
    
    # Sort ready operations by remaining work (descending) - process urgent ones first
    ops_sorted = sorted(enumerate(ready_operations), 
                        key=lambda x: x[1].get("remaining_work", 0), 
                        reverse=True)
    
    for idx, op in ops_sorted:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        # Find machine with minimum (current_load + duration)
        best_machine = None
        best_completion = float('inf')
        for machine, duration in mo.items():
            completion = sim_loads.get(machine, 0.0) + duration
            if completion < best_completion:
                best_completion = completion
                best_machine = machine
        
        if best_machine is not None:
            sim_loads[best_machine] = best_completion
    
    return sim_loads

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Check time budget
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        return _fast_dispatch(ready_operations, time)
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Compute greedy assignment to estimate future machine loads
    estimated_loads = _compute_greedy_assignment(ready_operations)
    
    # Find the operation that, when started on its best machine, 
    # minimizes the maximum machine completion time (load balancing)
    # Combined with urgency (remaining work)
    
    max_rw = max(op.get("remaining_work", 0) for op in ready_operations)
    if max_rw == 0:
        max_rw = 1.0
    
    best_score = -1
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        rw = op.get("remaining_work", 0)
        
        # For this operation, find the best machine considering load balancing
        # Score: prefer machines with lower estimated load, weighted by urgency
        best_machine_score = -1
        best_machine = None
        
        for machine, duration in mo.items():
            est_load = estimated_loads.get(machine, 0.0)
            
            # Load balancing score: prefer lower load
            # Urgency score: prefer higher remaining work
            urgency = rw / max_rw
            
            # Combined: prioritize urgent ops on least loaded machines
            # Use inverse load to penalize congested machines
            machine_score = urgency * (1.0 / (est_load + duration + 1.0))
            
            if machine_score > best_machine_score:
                best_machine_score = machine_score
                best_machine = machine
        
        if best_machine is None:
            continue
        
        # Final score for this operation: urgency + load balance benefit
        # Operations that can be placed on less congested machines get bonus
        load_balance_bonus = best_machine_score
        
        # Also consider: if this op has many options, it's more flexible
        n_opts = len(mo)
        flex_bonus = (n_opts - 1) / (n_opts + 1) if n_opts > 1 else 0.0
        
        score = load_balance_bonus + 0.1 * flex_bonus
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer higher remaining work
            if ready_operations[i].get("remaining_work", 0) > ready_operations[best_idx].get("remaining_work", 0):
                best_idx = i
    
    # Update machine loads: assume the chosen op will be started on its best machine
    # This helps subsequent dispatch calls account for this assignment
    chosen_op = ready_operations[best_idx]
    mo = chosen_op.get("machine_options", {})
    if mo:
        # Pick the machine with minimum estimated load + duration
        best_machine = None
        best_completion = float('inf')
        for machine, duration in mo.items():
            est_load = estimated_loads.get(machine, 0.0)
            completion = est_load + duration
            if completion < best_completion:
                best_completion = completion
                best_machine = machine
        
        if best_machine is not None:
            duration = mo[best_machine]
            _update_machine_load(best_machine, duration)
    
    return best_idx

# EVOLVE-BLOCK-END
