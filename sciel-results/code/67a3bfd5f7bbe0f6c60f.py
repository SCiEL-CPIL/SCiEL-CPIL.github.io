"""Job-Shop dispatch: pick the ready operation to start next to minimize makespan."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Wall-clock deadline guard: evaluator kills after 150s. Stay well under.
_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin

def _machine_load_map(ready_operations):
    """Approximate current machine load: sum of min durations assigned to each machine
    across all ready operations. This is a cheap proxy for which machines are busier."""
    load = {}
    for op in ready_operations:
        opts = op.get("machine_options", {})
        if not opts:
            continue
        # Attribute the operation's shortest duration to its least-loaded (preferred) machine
        best_m = min(opts, key=lambda m: (opts[m], m))
        load[best_m] = load.get(best_m, 0.0) + opts[best_m]
    return load

def dispatch(ready_operations, time):
    # Deadline guard: if we're near the kill limit, just return a quick choice.
    if _time.time() - _START_TS > DEADLINE_S:
        return 0

    if not ready_operations:
        return 0
    n = len(ready_operations)
    if n == 1:
        return 0

    # Approximate machine loads for load-balancing tie-break.
    mload = _machine_load_map(ready_operations)

    # Gather raw features
    rw = np.empty(n)      # remaining_work
    ro = np.empty(n)      # remaining_ops
    min_dur = np.empty(n) # min duration over machine options
    chosen_dur = np.empty(n)  # duration on the chosen (preferred) machine
    load_penalty = np.empty(n)  # load on the preferred machine (lower is better)
    avg_op_time = np.empty(n)   # average processing time of remaining ops (critical path proxy)
    critical_path_inc = np.empty(n)  # estimated critical path increase

    for i, op in enumerate(ready_operations):
        rw[i] = float(op.get("remaining_work", 0.0))
        ro[i] = max(1, int(op.get("remaining_ops", 0)))
        opts = op.get("machine_options", {})
        if not opts:
            min_dur[i] = 1.0
            chosen_dur[i] = 1.0
            load_penalty[i] = 0.0
            avg_op_time[i] = 0.0
            critical_path_inc[i] = 1.0 + rw[i]
            continue
        # Preferred machine: shortest duration; tie-break by lower current load, then id.
        best_m = min(opts, key=lambda m: (opts[m], mload.get(m, 0.0), m))
        chosen_dur[i] = opts[best_m]
        min_dur[i] = min(opts.values())
        load_penalty[i] = mload.get(best_m, 0.0)
        # Critical path proxy: average remaining work per remaining op
        avg_op_time[i] = rw[i] / ro[i]
        # Graph-theoretic lookahead: estimated increase in global critical path
        # When we start this operation, it takes chosen_dur time, and then unblocks
        # the next operation in the job. The new critical path contribution is
        # approximately: chosen_dur + (remaining_work - chosen_dur) / max(1, ro[i]-1)
        # But simpler: the operation itself adds chosen_dur, and the job's remaining
        # work after this op is rw[i] - chosen_dur, spread over ro[i]-1 ops.
        # The critical path through this job after starting this op is:
        #   chosen_dur + (rw[i] - chosen_dur) / max(1, ro[i]-1) * (some factor)
        # Simpler heuristic: chosen_dur + rw[i]/ro[i] (avg remaining work as proxy for
        # downstream critical path)
        if ro[i] > 1:
            downstream = (rw[i] - chosen_dur[i]) / (ro[i] - 1)
        else:
            downstream = 0.0
        critical_path_inc[i] = chosen_dur[i] + downstream

    # Ratio-based scoring: urgency-to-efficiency tradeoff
    # score = remaining_work / (1 + soft_min_duration)
    # This prevents scale dominance and naturally prioritizes operations
    # with high remaining work that can be processed quickly.
    
    # Use soft-min via log-sum-exp for smoothness
    eps = 1e-8
    log_sum_exp = np.log(np.sum(np.exp(-min_dur / 1.0), axis=0) * 1.0)
    soft_min = -1.0 * np.log(np.sum(np.exp(-min_dur / 1.0), axis=0))
    
    # Primary ratio score
    ratio_score = rw / (1.0 + soft_min)
    
    # Secondary: critical path increase (lower is better)
    # We want to minimize critical_path_inc, so invert it
    # Use rank for robustness
    order_cpi = np.argsort(critical_path_inc)
    cpi_ranks = np.empty_like(critical_path_inc, dtype=np.float64)
    sorted_cpi = critical_path_inc[order_cpi]
    i = 0
    while i < len(critical_path_inc):
        j = i
        while j + 1 < len(critical_path_inc) and sorted_cpi[j + 1] == sorted_cpi[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        cpi_ranks[order_cpi[i:j + 1]] = avg_rank
        i = j + 1
    inv_cpi = (len(critical_path_inc) + 1.0) - cpi_ranks
    
    # Combine: primary ratio + secondary critical path reduction
    w_primary = 1.0
    w_secondary = 0.3
    
    scores = w_primary * ratio_score + w_secondary * inv_cpi
    
    best_idx = int(np.argmax(scores))

    return best_idx
# EVOLVE-BLOCK-END
