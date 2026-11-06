"""
Job-Shop Scheduling Dispatch Rule
Uses a composite heuristic with load-aware and work-aware scoring.
"""
import numpy as np
import time as _time

# Global state for tracking machine loads
_machine_loads = {}
_total_time_elapsed = 0.0
_T0 = _time.time()
_BUDGET = 130.0  # Safety margin before 150s kill

def _get_min_duration(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return min(mo.values())

def _get_remaining_work(op):
    return op.get("remaining_work", 0.0)

def _get_remaining_ops(op):
    return op.get("remaining_ops", 0)

def _get_mean_duration(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return np.mean(list(mo.values()))

def _get_max_duration(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return max(mo.values())

def _get_spread(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    vals = list(mo.values())
    return max(vals) - min(vals)

def _get_n_options(op):
    mo = op.get("machine_options", {})
    return len(mo)

def dispatch(ready_operations, time):
    global _machine_loads, _total_time_elapsed
    
    # Time guard
    elapsed = _time.time() - _T0
    if elapsed > _BUDGET:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        # Update machine load for the single operation
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            dur = mo[best_machine]
            _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + dur
        _total_time_elapsed += _get_min_duration(op)
        return 0

    # Extract features
    remaining_works = np.array([_get_remaining_work(op) for op in ready_operations], dtype=float)
    min_durations = np.array([_get_min_duration(op) for op in ready_operations], dtype=float)
    mean_durations = np.array([_get_mean_duration(op) for op in ready_operations], dtype=float)
    max_durations = np.array([_get_max_duration(op) for op in ready_operations], dtype=float)
    remaining_ops = np.array([_get_remaining_ops(op) for op in ready_operations], dtype=float)
    spreads = np.array([_get_spread(op) for op in ready_operations], dtype=float)
    n_options = np.array([_get_n_options(op) for op in ready_operations], dtype=float)
    
    # Machine load features
    # For each operation, find the min-load machine among its options
    min_loads = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            min_loads[i] = min(_machine_loads.get(m, 0.0) for m in mo)
        else:
            min_loads[i] = 0.0

    # Normalize features to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    norm_work = normalize(remaining_works)
    norm_min_dur = normalize(min_durations)
    norm_mean_dur = normalize(mean_durations)
    norm_max_dur = normalize(max_durations)
    norm_ops = normalize(remaining_ops)
    norm_spread = normalize(spreads)
    norm_n_opts = normalize(n_options)
    norm_load = normalize(min_loads)

    # Composite Score
    # Priorities:
    # 1. High remaining work (MWR) - ensures long jobs start early
    # 2. Short processing time (SPT) - reduces idle time
    # 3. High remaining ops (MOR) - ensures jobs with many steps start early
    # 4. Low machine load - balances load across machines
    # 5. Low spread - prefer operations with consistent durations
    # 6. More options - flexibility is good

    w_work = 1.2
    w_min_dur = 0.8
    w_mean_dur = 0.3
    w_max_dur = 0.2
    w_ops = 0.5
    w_spread = 0.1
    w_n_opts = 0.1
    w_load = 0.6

    scores = (
        w_work * norm_work +
        w_min_dur * (1.0 - norm_min_dur) +
        w_mean_dur * (1.0 - norm_mean_dur) +
        w_max_dur * (1.0 - norm_max_dur) +
        w_ops * norm_ops +
        w_spread * (1.0 - norm_spread) +
        w_n_opts * norm_n_opts +
        w_load * (1.0 - norm_load)
    )

    # Select operation with highest score
    best_idx = int(np.argmax(scores))
    
    # Update machine load for the selected operation
    op = ready_operations[best_idx]
    mo = op.get("machine_options", {})
    if mo:
        # Pick the machine with minimum current load among options
        best_machine = min(mo, key=lambda m: _machine_loads.get(m, 0.0))
        dur = mo[best_machine]
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + dur
    
    _total_time_elapsed += _get_min_duration(op)
    
    return best_idx
