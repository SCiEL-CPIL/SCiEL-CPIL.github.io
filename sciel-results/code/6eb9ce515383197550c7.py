"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time and keep a safety margin.
_START_TIME = _time.perf_counter()
_DEADLINE = 150.0 - 5.0  # 5s safety margin


def _elapsed_ok():
    return (_time.perf_counter() - _START_TIME) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine runs it.
    
    Strategy: Combine multiple heuristics:
    1. Shortest processing time on the chosen machine (minimize immediate load)
    2. Work-to-time ratio (prioritize jobs with more remaining work per unit time)
    3. Balance machine loads by preferring machines with less total assigned work
    
    Returns (index, machine) to explicitly select the machine as well.
    """
    if not _elapsed_ok():
        # Safety: fall back to simple argmax
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration
        machine_options = op["machine_options"]
        best_machine = min(machine_options, key=lambda m: machine_options[m])
        return (0, best_machine)
    
    # Pre-compute features for each ready operation
    # For each op, we consider all machine options and compute a composite score
    
    # Track machine loads to balance work across machines
    machine_loads = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_loads[m] = machine_loads.get(m, 0) + 1
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        op_num = op["op"]
        machine_options = op["machine_options"]
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", sum(machine_options.values()))
        
        # For each eligible machine, compute a score
        for machine, duration in machine_options.items():
            # Heuristic 1: Shortest processing time with load balancing
            # Penalize machines that already have many operations queued
            load_penalty = machine_loads.get(machine, 0) * 0.1
            spf_score = -(duration + load_penalty)
            
            # Heuristic 2: Work-to-time ratio (higher remaining work per remaining op is better)
            if remaining_ops > 0:
                work_ratio = remaining_work / remaining_ops
            else:
                work_ratio = remaining_work
            
            # Heuristic 3: Remaining work (higher is better for making progress)
            rw_score = remaining_work
            
            # Heuristic 4: Machine availability (prefer less loaded machines)
            # Use a smooth penalty based on relative load
            if machine_loads:
                max_load = max(machine_loads.values()) if machine_loads else 1
                min_load = min(machine_loads.values()) if machine_loads else 0
                load_range = max_load - min_load if max_load > min_load else 1
                relative_load = (machine_loads.get(machine, 0) - min_load) / load_range
                availability_score = -relative_load * 0.5
            else:
                availability_score = 0.0
            
            # Composite score using smooth weighted combination
            # We want to prioritize:
            # - Jobs with high remaining work (to avoid starvation)
            # - Good work-to-time ratio
            # - Shorter processing time on less loaded machines
            score = (
                0.25 * rw_score +          # remaining work
                0.25 * work_ratio +         # work-to-time ratio
                0.35 * spf_score +          # shortest processing time with load penalty
                0.15 * availability_score   # machine availability
            )
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = machine
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
