"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = None

def _soft_min(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick."""
    if not values:
        return 0.0
    vals = list(values)
    m = min(vals)
    if m == 0:
        return 0.0
    # soft-min = -temperature * log(sum(exp(-v/temperature)))
    # For stability, shift by m: soft_min ≈ m - temperature * log(sum(exp(-(v-m)/temperature)))
    exp_terms = [math.exp(-(v - m) / temperature) for v in vals]
    s = sum(exp_terms)
    if s == 0:
        return m
    return m - temperature * math.log(s)

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Safety guard: if we are approaching the 120s limit, fall back to a fast heuristic
    # to ensure we don't timeout. The deadline is 120s, we check against 115s.
    if _time.time() - _START_TIME > 115:
        # Fast fallback: pick the operation with the shortest duration on its preferred machine
        # to keep the system moving.
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            # Find min duration for this op
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Use soft-min to capture the distribution of machine speeds, not just the best.
    # A lower soft-min means the operation has good options (low duration) across machines.
    
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += d

    best_score = -1
    best_idx = 0

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        # Use soft-min for duration to get a smoother signal
        durations = list(op["machine_options"].values())
        sm_dur = _soft_min(durations, temperature=0.5)
        
        # Estimate the load on machines this op can use
        loads = [machine_load[m] for m in op["machine_options"]]
        sm_load = _soft_min(loads, temperature=0.5)
        
        # Score: High remaining work is good, low soft-min duration is good, low machine load is good.
        # Weighted combination: prioritize urgency (rw) but penalize time cost and congestion.
        # Using a ratio form to balance scales.
        denom = sm_dur + sm_load + 1
        score = rw / denom
        
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
