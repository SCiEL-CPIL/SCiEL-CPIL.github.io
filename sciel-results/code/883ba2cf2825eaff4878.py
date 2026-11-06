"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


# Deadline guard: keep total dispatch calls well within budget.
_start_time = None
_CALL_COUNT = 0
_DEADLINE = 100.0  # seconds


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.monotonic()
    return _time.monotonic() - _start_time


def _pick_op(ready_operations):
    """
    Dispatch rule: Structural change to 'Contention-Aware Efficiency'.
    
    Instead of a linear combination of SPT and Work, we model the decision
    based on:
    1. Efficiency: Remaining Work / Min Duration (WSPT-like).
    2. Rigidity: Inverse of the Coefficient of Variation of machine durations.
       - High CV (flexible) -> Lower priority (can be scheduled later on other machines).
       - Low CV (rigid) -> Higher priority (bottleneck, must be scheduled when possible).
       
    Score = (Remaining Work / Min Duration) * (1 / (1 + CV))
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    scores = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        durations = np.array(list(op["machine_options"].values()), dtype=np.float64)
        
        if len(durations) == 0:
            continue
            
        min_dur = np.min(durations)
        mean_dur = np.mean(durations)
        
        # Avoid division by zero
        if min_dur == 0:
            min_dur = 1e-9
        if mean_dur == 0:
            mean_dur = 1e-9

        rem_work = op["remaining_work"]
        
        # Calculate Coefficient of Variation (CV)
        std_dur = np.std(durations)
        cv = std_dur / mean_dur
        
        # Efficiency component: WSPT
        # Higher remaining work relative to processing time is better
        efficiency = rem_work / min_dur
        
        # Rigidity component: 
        # We want to prioritize rigid operations (low CV).
        # Factor = 1 / (1 + CV). If CV=0, factor=1. If CV is high, factor drops.
        rigidity_factor = 1.0 / (1.0 + cv)
        
        # Combined Score
        # We prioritize high efficiency and high rigidity.
        scores[i] = efficiency * rigidity_factor
        
        # Add a small tie-breaker: prefer shorter min_dur if scores are close?
        # Or prefer more remaining ops?
        # Let's stick to the core structural change.
        
    return int(np.argmax(scores))


def dispatch(ready_operations, time):
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    return _pick_op(ready_operations)
# EVOLVE-BLOCK-END
