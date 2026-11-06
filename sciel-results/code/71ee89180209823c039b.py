"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 115.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy: Greedy bottleneck-elimination
    1. For each ready operation, compute its critical path length as
       remaining_ops * min_duration across its machine options.
    2. Select the operation with the maximum critical path length,
       prioritizing jobs that will otherwise dominate the makespan.
    3. Assign the selected operation to the machine with the lowest
       current load, approximated by the minimum duration among
       available options for that machine (lower duration = less load).
    
    The key insight is to eliminate the most constraining job first,
    preventing it from becoming a bottleneck later in the schedule.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)
    
    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    # Precompute per-operation features
    critical_paths = np.empty(n, dtype=float)
    min_durations = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        ro = float(op.get("remaining_ops", 1.0))
        opts = op["machine_options"]
        
        # Minimum duration for this operation
        if opts:
            min_d = min(opts.values())
        else:
            min_d = 0.0
        
        min_durations[i] = min_d
        
        # Critical path length: remaining_ops * min_duration
        # This estimates how long this job's critical path will be
        critical_paths[i] = ro * min_d
    
    # Find the operation with the maximum critical path length
    # This is the job that will dominate the makespan if not scheduled soon
    max_cp_idx = int(np.argmax(critical_paths))
    
    # For the selected operation, choose the machine with lowest load
    # Load is approximated by: duration on that machine * contention factor
    # Lower duration and lower contention = lower load
    selected_op = ready_operations[max_cp_idx]
    opts = selected_op["machine_options"]
    
    best_machine = None
    best_load = float('inf')
    
    for m, d in opts.items():
        d = float(d)
        if d <= 0:
            continue
        
        # Machine contention penalty: more ops competing = higher load
        count = machine_count.get(m, 1)
        # Load estimate: duration scaled by contention
        load = d * (1.0 + 0.3 * (count - 1))
        
        if load < best_load:
            best_load = load
            best_machine = m
    
    # Fallback if no valid machine was found
    if best_machine is None:
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            return max_cp_idx
    
    return (max_cp_idx, best_machine)
# EVOLVE-BLOCK-END
