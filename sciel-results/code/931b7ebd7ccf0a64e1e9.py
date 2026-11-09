"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic with soft-min surrogate and critical-path awareness.
    """
    if not ready_operations:
        return 0
    
    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Soft-min surrogate for machine durations using log-sum-exp
    # This captures the full distribution, not just the minimum
    # For each op, compute a soft-min of its machine_options values
    soft_min_durations = np.zeros(n, dtype=np.float64)
    min_durations = np.zeros(n, dtype=np.float64)
    max_durations = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            vals = np.array(list(machine_options.values()), dtype=np.float64)
            if len(vals) > 0:
                min_durations[i] = np.min(vals)
                max_durations[i] = np.max(vals)
                # Soft-min: -logsumexp(-x) ≈ min(x) but smooth
                # Use temperature parameter for control
                # Lower temp emphasizes the minimum more strongly
                temp = 0.5
                shifted = -vals / temp
                max_shifted = np.max(shifted)
                if max_shifted > -1e10:
                    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
                    soft_min_durations[i] = -temp * log_sum_exp
                else:
                    soft_min_durations[i] = np.min(vals)
            else:
                soft_min_durations[i] = 0.0
                min_durations[i] = 0.0
                max_durations[i] = 0.0
        else:
            soft_min_durations[i] = 0.0
            min_durations[i] = 0.0
            max_durations[i] = 0.0
    
    # Critical path estimate: remaining_work + current_op_duration
    # The operation that is on the critical path should be prioritized
    critical_path_scores = remaining_works + soft_min_durations
    
    # Machine contention proxy: if many ops share similar min durations,
    # we want to prioritize the one with the most remaining work to clear it
    # Compute a "scarcity" factor: how unique is this op's duration profile
    # Use the spread (max - min) as an indicator of flexibility
    spreads = max_durations - min_durations
    
    # Normalize each feature
    max_mw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_cp = np.max(critical_path_scores) if np.max(critical_path_scores) > 0 else 1.0
    max_mor = np.max(remaining_ops_counts) if np.max(remaining_ops_counts) > 0 else 1.0
    max_sm = np.max(soft_min_durations) if np.max(soft_min_durations) > 0 else 1.0
    max_spread = np.max(spreads) if np.max(spreads) > 0 else 1.0
    
    normalized_mw = remaining_works / max_mw
    normalized_cp = critical_path_scores / max_cp
    normalized_mor = remaining_ops_counts / max_mor
    normalized_sm = soft_min_durations / max_sm
    normalized_spread = spreads / max_spread
    
    # Combine with tuned weights:
    # Critical path (0.40): most important - operations that determine makespan
    # Remaining work (0.25): secondary indicator
    # Soft-min duration (0.15): processing time aspect
    # Remaining ops (0.10): tiebreaker for sequence length
    # Spread (0.10): prefer ops with LESS flexibility (smaller spread)
    # because they have fewer machine options and might block
    normalized_spread_inv = 1.0 - normalized_spread
    
    combined_scores = (
        0.40 * normalized_cp +
        0.25 * normalized_mw +
        0.15 * normalized_sm +
        0.10 * normalized_mor +
        0.10 * normalized_spread_inv
    )
    
    # Return index of the operation with the highest combined score
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
