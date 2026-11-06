"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


# Global state to cache parameters and track time
_start_time = _time.time()
_DEADLINE = 150.0  # Safety margin before 150s kill

# Track machine loads to estimate contention
_machine_loads = {}  # machine_id -> estimated remaining load

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, use a very fast fallback
    elapsed = _time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: pick the operation with minimum duration
        best_index = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            machine_options = op.get("machine_options", {})
            if not machine_options:
                continue
            md = min(machine_options.values())
            if md < min_dur:
                min_dur = md
                best_index = i
        return int(best_index)

    # Dynamic resource contention model:
    # For each ready operation, compute an "effective processing time" for each
    # eligible machine that accounts for current machine load (contention).
    # Priority score balances:
    #   - Critical path urgency (remaining work)
    #   - Constraint tightness (few machine options)
    #   - Contention-aware efficiency (lower effective time on less-loaded machines)
    #
    # We select the operation with the highest priority score, and if returning
    # a machine, pick the one minimizing predicted completion time.
    
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        num_options = len(machine_options)
        remaining_work = op.get("remaining_work", 0)
        
        # Compute contention-aware effective processing times
        # For each machine, effective_time = duration + alpha * (machine_load / num_options)
        # where machine_load is our estimate of how busy that machine is
        min_effective = float('inf')
        best_mach_for_op = None
        alpha = 0.3  # Weight for contention penalty
        
        for m, dur in machine_options.items():
            load = _machine_loads.get(m, 0.0)
            # Effective time includes a contention penalty proportional to current load
            eff_time = dur + alpha * (load / max(num_options, 1))
            if eff_time < min_effective:
                min_effective = eff_time
                best_mach_for_op = m
        
        # Also track raw min duration for tightness/speed bonus
        min_duration = min(machine_options.values())
        
        # Priority score components:
        # 1. Critical path urgency: remaining_work / num_options (weighted LPT)
        # 2. Constraint tightness: 1 / num_options
        # 3. Contention-aware efficiency: 1 / min_effective (prefer ops that can run fast on less-loaded machines)
        # 4. Speed bonus: 1 / min_duration
        
        if remaining_work == 0:
            urgency_term = 0.0
        else:
            urgency_term = remaining_work / num_options
        
        tightness_term = 1.0 / num_options
        efficiency_term = 1.0 / max(min_effective, 1e-9)
        speed_term = 1.0 / max(min_duration, 1e-9)
        
        # Combine: urgency dominates, then efficiency, then tightness, then speed
        score = urgency_term + 2.0 * efficiency_term + tightness_term + 0.5 * speed_term
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_mach_for_op
    
    # Update machine loads: add the duration of the selected operation to its machine
    if best_machine is not None:
        selected_op = ready_operations[best_index]
        selected_dur = selected_op.get("machine_options", {}).get(best_machine, 0)
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + selected_dur
    
    return int(best_index)
# EVOLVE-BLOCK-END
