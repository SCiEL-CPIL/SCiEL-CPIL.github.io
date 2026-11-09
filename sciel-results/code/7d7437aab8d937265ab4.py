"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.monotonic()
_DEADLINE = 115.0
_SAFETY_MARGIN = 5.0

# Stateful machine availability tracking
_machine_free_time = {}
# Track per-machine load for bottleneck awareness
_machine_load = {}


def dispatch(ready_operations, time):
    global _machine_free_time, _machine_load
    
    # Check deadline guard
    if _time.monotonic() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with maximum remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Use soft-min for smooth surrogate of minimum duration
    # and soft-max for smooth surrogate of maximum remaining work
    
    rw_list = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    rem_ops_list = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=float)
    
    # Soft-min for processing times (lower is better, so we want to minimize)
    # Use log-sum-exp trick for numerical stability
    # soft_min(x) = -logsumexp(-x)
    def soft_min(x, tau=1.0):
        if len(x) == 0:
            return 0.0
        neg_x = -x / tau
        max_neg_x = np.max(neg_x)
        if max_neg_x > 700:  # numerical safety
            return float(np.min(x))
        log_sum = np.log(np.sum(np.exp(neg_x - max_neg_x))) + max_neg_x
        return -log_sum * tau
    
    # Compute soft-min for each operation's machine options
    # This gives a smooth estimate of the minimum processing time
    # Using a smaller tau for a sharper approximation of the minimum
    soft_min_durs = np.array([soft_min(np.array(list(op["machine_options"].values()), dtype=float), tau=0.1) for op in ready_operations])
    
    # For each operation, compute earliest finish time considering machine availability
    # This is a key improvement: use stateful machine tracking
    earliest_finish = np.zeros(n, dtype=float)
    best_machine_for_op = [None] * n
    
    for i, op in enumerate(ready_operations):
        best_finish = float('inf')
        best_mach = None
        for mach, dur in op["machine_options"].items():
            mach_free = _machine_free_time.get(mach, 0.0)
            start_time = max(time, mach_free)
            finish_time = start_time + dur
            if finish_time < best_finish:
                best_finish = finish_time
                best_mach = mach
        earliest_finish[i] = best_finish
        best_machine_for_op[i] = best_mach
    
    # SPT term: 1/duration favors short operations on specific machines (local efficiency)
    # Use reciprocal with a small epsilon to avoid division by zero
    eps = 1e-6
    spt_term = 1.0 / (soft_min_durs + eps)
    
    # Earliest finish time term: lower is better (so we invert for scoring)
    # Normalize earliest finish times
    ef_range = np.max(earliest_finish) - np.min(earliest_finish) if np.max(earliest_finish) > np.min(earliest_finish) else 1.0
    ef_norm = (earliest_finish - np.min(earliest_finish)) / ef_range
    
    # Combine scores with smooth surrogates
    # Balanced strategy: global job urgency (remaining work) + local machine efficiency (SPT) + earliest finish + bottleneck awareness
    
    # Normalize components
    rw_range = np.max(rw_list) - np.min(rw_list) if np.max(rw_list) > np.min(rw_list) else 1.0
    dur_range = np.max(soft_min_durs) - np.min(soft_min_durs) if np.max(soft_min_durs) > np.min(soft_min_durs) else 1.0
    ops_range = np.max(rem_ops_list) - np.min(rem_ops_list) if np.max(rem_ops_list) > np.min(rem_ops_list) else 1.0
    spt_range = np.max(spt_term) - np.min(spt_term) if np.max(spt_term) > np.min(spt_term) else 1.0
    
    rw_norm = (rw_list - np.min(rw_list)) / rw_range
    dur_norm = (soft_min_durs - np.min(soft_min_durs)) / dur_range
    ops_norm = (rem_ops_list - np.min(rem_ops_list)) / ops_range
    spt_norm = (spt_term - np.min(spt_term)) / spt_range
    
    # Score: balance global urgency (remaining work), local efficiency (SPT), earliest finish, and bottleneck awareness
    # SPT term gets strong weight to prioritize short operations on specific machines
    # Remaining work provides global job completion pressure
    # Earliest finish time is inverted (we want lower finish times, so subtract from 1)
    # Bottleneck (high duration) term provides secondary awareness for long operations
    scores = rw_norm * 0.7 + spt_norm * 1.0 + (1.0 - ef_norm) * 1.2 + dur_norm * 0.2 + ops_norm * 0.2
    
    best_idx = int(np.argmax(scores))
    
    # Update machine state with the chosen operation
    if best_machine_for_op[best_idx] is not None:
        chosen_mach = best_machine_for_op[best_idx]
        chosen_dur = ready_operations[best_idx]["machine_options"][chosen_mach]
        _machine_free_time[chosen_mach] = earliest_finish[best_idx]
        _machine_load[chosen_mach] = _machine_load.get(chosen_mach, 0.0) + chosen_dur
    
    return best_idx
# EVOLVE-BLOCK-END
