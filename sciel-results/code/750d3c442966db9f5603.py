"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def dispatch(ready_operations, time):
    # Check deadline guard
    if _time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with maximum minimum processing time (LPT)
        min_durations = [min(op["machine_options"].values()) for op in ready_operations]
        return int(np.argmax(min_durations))
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features
    rws = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    min_durs = np.array([min(op["machine_options"].values()) if op["machine_options"] else 0 for op in ready_operations], dtype=np.float64)
    rem_ops = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=np.float64)
    
    # Normalize features to [0, 1] range
    rw_min, rw_max = rws.min(), rws.max()
    dur_min, dur_max = min_durs.min(), min_durs.max()
    ops_min, ops_max = rem_ops.min(), rem_ops.max()
    
    if rw_max > rw_min:
        rw_norm = (rws - rw_min) / (rw_max - rw_min)
    else:
        rw_norm = np.ones(n)
    
    if dur_max > dur_min:
        dur_norm = (min_durs - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.ones(n)
    
    if ops_max > ops_min:
        ops_norm = (rem_ops - ops_min) / (ops_max - ops_min)
    else:
        ops_norm = np.ones(n)
    
    # Greedy Bottleneck Elimination: prioritize LONG operations (LPT)
    # to prevent them from becoming critical path bottlenecks later
    # Combined with remaining work (MWR) and urgency (remaining ops)
    
    # Primary: LPT - longer min_duration gets higher priority (bottleneck elimination)
    # Secondary: MWR - more remaining work gets priority (avoid starvation)
    # Tertiary: Fewer remaining ops gets slight priority (completion urgency)
    
    w_lpt = 0.55
    w_mrw = 0.35
    w_ops = 0.10
    
    # LPT: higher dur_norm is better (longer operations scheduled first)
    # MWR: higher rw_norm is better (more work remaining)
    # Ops: LOWER ops_norm is better (fewer remaining ops = closer to completion)
    scores = w_lpt * dur_norm + w_mrw * rw_norm - w_ops * ops_norm
    
    # Add a small tie-breaker based on index for determinism
    scores += np.arange(n) * 1e-10
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
