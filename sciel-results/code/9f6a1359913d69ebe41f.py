"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict


# Module-level deadline guard
_START_TIME = None
_TIME_LIMIT = 135.0  # 15s safety margin before 150s kill
_DEADLINE_REACHED = False

# State for load-aware dispatch
_machine_loads = defaultdict(float)
_dispatch_count = 0


def _check_deadline():
    global _START_TIME, _DEADLINE_REACHED
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    if _time.monotonic() - _START_TIME > _TIME_LIMIT:
        _DEADLINE_REACHED = True
    return _DEADLINE_REACHED


def _reset_state():
    global _machine_loads, _dispatch_count
    _machine_loads = defaultdict(float)
    _dispatch_count = 0


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling with load-aware machine assignment.
    
    Strategy: Return (index, machine) to explicitly assign each operation to a machine.
    Machine selection balances:
    - Processing time on that machine (shorter is better)
    - Current load on that machine (less loaded is better)
    
    Operation selection score combines:
    1. Remaining work (total) - prefer jobs with more remaining work
    2. Minimum machine option duration via soft-min - smoother surrogate
    3. Number of machine options (flexibility) - prefer operations with more choices
    4. Remaining ops - prefer jobs with fewer remaining ops (to complete jobs)
    5. Load-aware adjustment: prefer operations that can be placed on less-loaded machines
    """
    global _machine_loads, _dispatch_count
    
    if not ready_operations:
        return 0
    
    if _check_deadline():
        # Fallback: return first operation with its fastest machine
        op = ready_operations[0]
        mo = op["machine_options"]
        if isinstance(mo, dict) and mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0
    
    n = len(ready_operations)
    
    # Compute scores using numpy for efficiency
    scores = np.zeros(n)
    best_machine_for_op = [None] * n
    best_load_adjusted_cost = [0.0] * n
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get machine options
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict) and machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            machine_keys = list(machine_options.keys())
            n_machines = len(machine_options)
            
            # Soft-min via log-sum-exp: approximates min(durations) smoothly
            temp = 0.5
            log_sum_exp = np.log(np.sum(np.exp(-durations / temp))) - np.log(n_machines)
            soft_min = -temp * log_sum_exp
            
            # Flexibility bonus: more machine options means more scheduling flexibility
            flexibility = np.log1p(n_machines)
            
            # Load-aware machine selection:
            # For each machine option, compute effective cost = duration + alpha * current_load
            # Choose the machine with minimum effective cost
            alpha = 0.5  # Weight for load balancing
            effective_costs = []
            for m in machine_keys:
                dur = machine_options[m]
                load = _machine_loads.get(m, 0.0)
                effective_costs.append(dur + alpha * load)
            
            effective_costs = np.array(effective_costs, dtype=np.float64)
            best_m_idx = int(np.argmin(effective_costs))
            best_machine = machine_keys[best_m_idx]
            best_load_adjusted_cost[i] = effective_costs[best_m_idx]
            best_machine_for_op[i] = best_machine
            
            # Load-aware score adjustment: prefer ops that can be placed on less-loaded machines
            # Lower effective cost is better
            load_adjustment = -best_load_adjusted_cost[i] * 0.3
        else:
            soft_min = 0.0
            flexibility = 0.0
            best_machine = None
            load_adjustment = 0.0
            best_machine_for_op[i] = None
            best_load_adjusted_cost[i] = 0.0
        
        # Weighted score: 
        # - Emphasize remaining work (keep busy jobs progressing)
        # - Penalize long minimum duration (shorter ops are easier to slot in)
        # - Reward flexibility (more machine options = more scheduling options)
        # - Slightly favor jobs with fewer remaining ops (complete jobs sooner)
        # - Load-aware adjustment: prefer ops that can be placed on less-loaded machines
        scores[i] = (
            remaining_work * 1.0
            - soft_min * 0.4
            + flexibility * 0.3
            - remaining_ops * 0.2
            + load_adjustment
        )
    
    best_idx = int(np.argmax(scores))
    
    # Update machine load for the selected operation
    op = ready_operations[best_idx]
    mo = op["machine_options"]
    if isinstance(mo, dict) and mo:
        best_machine = best_machine_for_op[best_idx]
        if best_machine is None:
            best_machine = min(mo, key=mo.get)
        
        # Update the load for this machine
        _machine_loads[best_machine] += mo[best_machine]
        _dispatch_count += 1
        
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
