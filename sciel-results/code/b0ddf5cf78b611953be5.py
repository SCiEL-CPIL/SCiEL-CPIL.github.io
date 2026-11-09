"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global state for tracking machine availability across dispatch calls
# This allows us to estimate completion times more accurately than just using min_duration
_machine_free_times = {}
_instance_start_time = None
_last_time_seen = -1.0
_dispatch_count = 0

# Deadline guard state
_wall_start = None
_DEADLINE_LIMIT = 145.0  # Safety margin from 150s limit

def _reset_state():
    global _machine_free_times, _instance_start_time, _last_time_seen, _dispatch_count
    _machine_free_times = {}
    _instance_start_time = None
    _last_time_seen = -1.0
    _dispatch_count = 0

def dispatch(ready_operations, time):
    global _machine_free_times, _instance_start_time, _last_time_seen, _dispatch_count, _wall_start
    
    # Detect new instance (time goes backwards or significantly resets)
    if time < _last_time_seen - 1e-6:
        _reset_state()
    _last_time_seen = time
    
    if _wall_start is None:
        _wall_start = _time.time()
    
    # Deadline guard: if we're running too long, fall back to a very fast heuristic
    if _time.time() - _wall_start > _DEADLINE_LIMIT:
        # Fast fallback: SPT on earliest available machine
        best_idx = 0
        best_completion = float('inf')
        for i, op in enumerate(ready_operations):
            options = op.get("machine_options", {})
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if min_completion < best_completion:
                best_completion = min_completion
                best_idx = i
        return int(best_idx)
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        idx = 0
        op = ready_operations[0]
        options = op.get("machine_options", {})
        if options:
            best_m = min(options.keys(), key=lambda m: max(time, _machine_free_times.get(m, 0)) + options[m])
            dur = options[best_m]
            start_t = max(time, _machine_free_times.get(best_m, 0))
            finish_t = start_t + dur
            _machine_free_times[best_m] = finish_t
        return idx
    
    _dispatch_count += 1
    
    # Compute scores using a hybrid approach:
    # 1. Estimated completion time (considers machine availability) - primary
    # 2. SPT (1/min_duration) - secondary  
    # 3. MWR (remaining work) - tertiary
    
    scores = np.zeros(n)
    completion_times = np.full(n, float('inf'))
    min_durations = np.zeros(n)
    remaining_works = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        options = op.get("machine_options", {})
        if not options:
            min_durations[i] = 1.0
            remaining_works[i] = op.get("remaining_work", 0)
            completion_times[i] = float('inf')
            continue
        
        # Find best machine for this operation (earliest completion)
        min_completion = float('inf')
        best_m_for_op = None
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            if completion_t < min_completion:
                min_completion = completion_t
                best_m_for_op = m
        
        min_durations[i] = min(options.values())
        remaining_works[i] = op.get("remaining_work", 0)
        completion_times[i] = min_completion if best_m_for_op is not None else float('inf')
    
    # Avoid division by zero
    min_durations_safe = min_durations + 1e-6
    completion_times_safe = np.where(np.isfinite(completion_times), completion_times, 1e6) + 1e-6
    
    # Component 1: Estimated completion time (lower is better) -> invert for max
    # Use 1/completion_time to normalize
    completion_scores = 1.0 / completion_times_safe
    
    # Component 2: SPT (shorter min duration is better)
    spt_scores = 1.0 / min_durations_safe
    
    # Component 3: MWR (more remaining work is better)
    rw_min = np.min(remaining_works)
    rw_max = np.max(remaining_works)
    if rw_max > rw_min:
        mrw_scores = (remaining_works - rw_min) / (rw_max - rw_min)
    else:
        mrw_scores = np.zeros(n)
    
    # Combine with weights:
    # - Completion time is the most direct proxy for makespan reduction
    # - SPT helps clear machines quickly
    # - MWR prevents starvation of long jobs
    w_completion = 0.55
    w_spt = 0.35
    w_mrw = 0.10
    
    scores = w_completion * completion_scores + w_spt * spt_scores + w_mrw * mrw_scores
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    # Update global state with the chosen assignment
    sel_op = ready_operations[best_idx]
    options = sel_op.get("machine_options", {})
    if options:
        best_m = None
        min_completion = float('inf')
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            if completion_t < min_completion:
                min_completion = completion_t
                best_m = m
        if best_m is not None:
            dur = options[best_m]
            start_t = max(time, _machine_free_times.get(best_m, 0))
            finish_t = start_t + dur
            _machine_free_times[best_m] = finish_t
    
    return best_idx
# EVOLVE-BLOCK-END
