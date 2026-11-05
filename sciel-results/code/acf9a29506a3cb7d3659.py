"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_DEADLINE_SECONDS = 115.0  # safety margin below the 120s wall-clock kill
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.perf_counter()
    return _time.perf_counter() - _START_TIME


def _soft_min(values, temperature=0.5):
    """Smooth approximation to min using log-sum-exp (negative)."""
    if len(values) == 1:
        return values[0]
    vals = np.asarray(values, dtype=np.float64)
    min_val = np.min(vals)
    # Avoid overflow: subtract min before exp
    exps = np.exp((min_val - vals) / temperature)
    return min_val - temperature * math.log(np.sum(exps))


def _soft_max(values, temperature=0.5):
    """Smooth approximation to max using log-sum-exp."""
    if len(values) == 1:
        return values[0]
    vals = np.asarray(values, dtype=np.float64)
    max_val = np.max(vals)
    exps = np.exp((vals - max_val) / temperature)
    return max_val + temperature * math.log(np.sum(exps))


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to start next.
    
    Strategy: Combine multiple heuristics into a composite score.
    - Prefer operations with less remaining work (to finish jobs quickly)
    - Prefer operations that can be done on the least-loaded machine
    - Prefer operations with fewer remaining ops (to unlock downstream)
    - Use soft-min for machine selection to smooth the objective
    
    Returns: index of the ready operation to start (or (index, machine) tuple)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS:
        # Fallback: return the first operation
        return 0
    
    # Precompute features for each ready operation
    scores = []
    machine_choices = []
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        op_id = op["op"]
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        
        # Find the best machine (shortest duration)
        if machine_options:
            min_duration = min(machine_options.values())
            best_machines = [m for m, d in machine_options.items() if d == min_duration]
            # Use soft-min for a smooth machine score
            durations = list(machine_options.values())
            soft_min_duration = _soft_min(durations, temperature=0.3)
        else:
            min_duration = 0
            best_machines = []
            soft_min_duration = 0
        
        # Composite score: lower is better
        # 1. Remaining work: prefer less remaining work (finish jobs faster)
        work_score = remaining_work
        
        # 2. Number of remaining ops: prefer fewer (unlock downstream sooner)
        ops_score = remaining_ops * 1.0  # weight
        
        # 3. Machine flexibility: prefer operations with more machine options
        #    (more flexibility = less risk of bottleneck)
        n_options = len(machine_options) if machine_options else 1
        flexibility_score = -math.log(n_options + 1)  # negative so more options = lower score
        
        # 4. Soft-min duration as a proxy for machine load
        duration_score = soft_min_duration
        
        # Combine with weights
        # Emphasize finishing jobs with less remaining work
        composite = (
            1.0 * work_score +
            0.5 * ops_score +
            0.1 * flexibility_score +
            0.8 * duration_score
        )
        
        scores.append(composite)
        machine_choices.append(best_machines)
    
    # Pick the operation with the lowest composite score
    best_idx = int(np.argmin(scores))
    
    # Optionally return the best machine for that operation
    best_op = ready_operations[best_idx]
    if best_op["machine_options"]:
        min_dur = min(best_op["machine_options"].values())
        best_machines = [m for m, d in best_op["machine_options"].items() if d == min_dur]
        # Pick the first best machine (deterministic)
        best_machine = best_machines[0]
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
