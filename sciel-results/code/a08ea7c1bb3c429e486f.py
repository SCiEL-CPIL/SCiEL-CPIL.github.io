import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # Pre-calculate min duration for each operation
    # min_dur[i] is the shortest processing time for operation i
    min_durs = [0.0] * n_ops
    remaining_ops_list = [0] * n_ops
    remaining_work_list = [0.0] * n_ops

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            # Get the minimum duration across all machine options
            # This represents the fastest possible completion for this op
            min_durs[i] = min(machine_options.values())
        else:
            min_durs[i] = 1.0 # Fallback

        remaining_ops_list[i] = op.get("remaining_ops", 1)
        remaining_work_list[i] = op.get("remaining_work", 0.0)
        
        if remaining_ops_list[i] <= 0:
            remaining_ops_list[i] = 1
        if remaining_work_list[i] is None:
            remaining_work_list[i] = 0.0

    # Find the best operation
    # Primary: Shortest Processing Time (SPT) -> lower min_dur is better
    # Secondary: Fewer remaining operations -> lower remaining_ops is better
    # Tertiary: Less remaining work -> lower remaining_work is better
    
    best_index = 0
    best_min_dur = float('inf')
    best_rem_ops = float('inf')
    best_rem_work = float('inf')

    for i in range(n_ops):
        md = min_durs[i]
        ro = remaining_ops_list[i]
        rw = remaining_work_list[i]

        # Slight epsilon to handle floating point issues
        if md < best_min_dur - 1e-9:
            best_min_dur = md
            best_rem_ops = ro
            best_rem_work = rw
            best_index = i
        elif abs(md - best_min_dur) <= 1e-9:
            # Tie-breaker 1: Lower remaining ops
            if ro < best_rem_ops - 1e-9:
                best_rem_ops = ro
                best_rem_work = rw
                best_index = i
            elif abs(ro - best_rem_ops) <= 1e-9:
                # Tie-breaker 2: Lower remaining work
                if rw < best_rem_work - 1e-9:
                    best_rem_work = rw
                    best_index = i

    return int(best_index)
