"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0


def _is_deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)

    if n == 1:
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    ops = np.array([op["op"] for op in ready_operations], dtype=np.float64)

    # Use soft-min surrogate for smoother processing time estimation
    # This avoids the nonsmooth min() and provides a differentiable-like signal
    soft_min_temp = 0.5  # Temperature parameter for soft-min
    min_durations = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = np.array(list(mo.values()), dtype=np.float64)
            # Soft-min: -log(sum(exp(-x/T)))/T * T = -T*log(sum(exp(-x/T)))
            # For numerical stability, use the standard soft-min formula
            neg_exp = np.exp(-durations / soft_min_temp)
            min_durations[i] = -soft_min_temp * np.log(np.sum(neg_exp))
            # Clamp to reasonable range
            min_durations[i] = max(min_durations[i], 0.1)
        else:
            min_durations[i] = 0.0

    # Normalize features to [0, 1] range to prevent scale domination
    max_rw = remaining_works.max() if remaining_works.max() > 0 else 1.0
    max_ro = remaining_ops.max() if remaining_ops.max() > 0 else 1.0
    max_op = ops.max() if ops.max() > 0 else 1.0
    max_md = min_durations.max() if min_durations.max() > 0 else 1.0

    # Feature 1: WSPT ratio - remaining work per unit processing time
    # Higher = more work packed into this operation, should prioritize
    wspt_ratio = remaining_works / (min_durations + 1e-9)
    max_wspt = wspt_ratio.max() if wspt_ratio.max() > 0 else 1.0
    wspt_norm = wspt_ratio / max_wspt

    # Feature 2: Remaining work (normalized) - longer jobs get priority
    rw_norm = remaining_works / max_rw

    # Feature 3: Remaining ops (normalized) - more ops = keep in motion to avoid blocking
    ro_norm = remaining_ops / max_ro

    # Feature 4: Op number (normalized) - later operations are more critical
    op_norm = ops / max_op

    # Feature 5: SPT bonus - prefer shorter processing times to reduce idle
    spt_bonus = 1.0 / (min_durations + 1.0)
    max_spt = spt_bonus.max() if spt_bonus.max() > 0 else 1.0
    spt_norm = spt_bonus / max_spt

    # Feature 6: Machine contention proxy - operations with fewer machine options
    # are more constrained and should be prioritized
    n_machines = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    max_nm = n_machines.max() if n_machines.max() > 0 else 1.0
    # Invert: fewer machines = higher urgency
    constraint_norm = (max_nm - n_machines + 1.0) / (max_nm + 1.0)

    # Combine with tuned weights
    # Emphasize WSPT (work-per-time) and remaining work, with secondary signals
    # Shift weight from op_norm to constraint_norm which captures machine availability
    w_wspt = 0.30
    w_rw = 0.25
    w_ro = 0.15
    w_op = 0.05
    w_spt = 0.10
    w_constraint = 0.15

    scores = (
        w_wspt * wspt_norm
        + w_rw * rw_norm
        + w_ro * ro_norm
        + w_op * op_norm
        + w_spt * spt_norm
        + w_constraint * constraint_norm
    )

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
