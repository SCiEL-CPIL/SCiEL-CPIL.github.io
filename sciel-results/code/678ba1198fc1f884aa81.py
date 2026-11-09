"""Job-Shop dispatch: pick the ready operation to start next to minimize makespan."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Wall-clock deadline guard: evaluator kills after 150s. Stay well under.
_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin

# Stateful tracking for machine availability
_machine_free_times = {}
_last_time = -1.0
_dispatch_count = 0

def _reset_state_if_new_instance(time):
    """Detect if we are starting a new instance (time goes backwards or resets significantly)."""
    global _machine_free_times, _last_time, _dispatch_count
    # If time goes backwards, it's a new instance
    if time < _last_time - 1e-6:
        _machine_free_times = {}
        _dispatch_count = 0
    _last_time = time

def dispatch(ready_operations, time):
    global _machine_free_times, _dispatch_count, _last_time
    
    _reset_state_if_new_instance(time)

    # Deadline guard: if we're near the kill limit, just return a quick choice.
    if _time.time() - _START_TS > DEADLINE_S:
        # Fallback: Fast SPT-like heuristic
        best_idx = 0
        best_key = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op.get("machine_options", {})
            if not opts:
                continue
            # Simple min duration
            min_d = min(opts.values())
            if min_d < best_key:
                best_key = min_d
                best_idx = i
        return best_idx

    if not ready_operations:
        return 0
    n = len(ready_operations)
    if n == 1:
        # Single op: just assign it and update state
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            # Pick the machine that finishes earliest
            best_m = min(opts, key=lambda m: max(time, _machine_free_times.get(m, 0.0)) + opts[m])
            dur = opts[best_m]
            start_t = max(time, _machine_free_times.get(best_m, 0.0))
            _machine_free_times[best_m] = start_t + dur
        return 0

    _dispatch_count += 1

    # Precompute per-operation features
    # We use stateful _machine_free_times to get accurate earliest finish times
    
    rw = np.empty(n)          # remaining_work
    ro = np.empty(n)          # remaining_ops
    est_finish = np.empty(n)  # earliest possible finish time if started now
    bottleneck_risk = np.empty(n) # how much this op extends the current max load
    
    current_max_load = max(_machine_free_times.values()) if _machine_free_times else time
    # The current makespan estimate is max(time, current_max_load)
    current_makespan_est = max(time, current_max_load)

    for i, op in enumerate(ready_operations):
        rw[i] = float(op.get("remaining_work", 0.0))
        ro[i] = max(1, int(op.get("remaining_ops", 0)))
        opts = op.get("machine_options", {})
        
        if not opts:
            est_finish[i] = time
            bottleneck_risk[i] = 0.0
            continue
        
        # Find the machine that allows this operation to finish earliest
        best_m = min(opts, key=lambda m: max(time, _machine_free_times.get(m, 0.0)) + opts[m])
        dur = opts[best_m]
        start_t = max(time, _machine_free_times.get(best_m, 0.0))
        finish_t = start_t + dur
        
        est_finish[i] = finish_t
        # Bottleneck risk: how much does this finish time exceed the current estimated makespan?
        # If finish_t > current_makespan_est, it increases the makespan.
        bottleneck_risk[i] = max(0.0, finish_t - current_makespan_est)

    # Robust normalization via ranks to avoid scale issues
    def rank(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(arr, dtype=np.float64)
        sorted_vals = arr[order]
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    # We want to FAVOR:
    #  - high remaining_work  -> higher rank is better (critical jobs first)
    #  - low est_finish       -> lower is better (SPT/EFFICIENCY: finish soon)
    #  - high remaining_ops   -> higher is better (more critical)
    #  - low bottleneck_risk  -> lower is better (avoid extending makespan)

    rw_rank = rank(rw)          # high = good
    ef_rank = rank(est_finish)  # high = late (bad), so invert
    ro_rank = rank(ro)          # high = many (good)
    br_rank = rank(bottleneck_risk) # high = risky (bad), so invert

    inv_ef = (len(rw) + 1.0) - ef_rank
    inv_br = (len(rw) + 1.0) - br_rank

    # Composite score
    # Weights tuned to balance criticality (MWRF) with efficiency (SPT) and balancing
    w_rw = 1.2
    w_ef = 1.0
    w_ro = 0.5
    w_br = 0.5

    scores = (w_rw * rw_rank
              + w_ef * inv_ef
              + w_ro * ro_rank
              + w_br * inv_br)

    best_idx = int(np.argmax(scores))

    # Update state with the chosen assignment
    sel_op = ready_operations[best_idx]
    opts = sel_op.get("machine_options", {})
    if opts:
        best_m = min(opts, key=lambda m: max(time, _machine_free_times.get(m, 0.0)) + opts[m])
        dur = opts[best_m]
        start_t = max(time, _machine_free_times.get(best_m, 0.0))
        finish_t = start_t + dur
        _machine_free_times[best_m] = finish_t

    return best_idx
# EVOLVE-BLOCK-END
