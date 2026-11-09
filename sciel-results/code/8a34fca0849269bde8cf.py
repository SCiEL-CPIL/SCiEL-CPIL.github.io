"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _elapsed() -> float:
    return _time.time() - _START_TIME


def _within_deadline() -> bool:
    return _elapsed() < 150.0 - _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: dynamic bottleneck identification. For each candidate operation,
    assign it to its least-loaded (shortest-duration) available machine and
    compute the resulting minimum slack across all ready operations.
    Select the operation whose assignment maximizes this minimum slack,
    thereby preventing machine starvation and reducing idle gaps.
    
    Uses a soft-min (log-sum-exp) surrogate for robustness, combined with
    hard urgency signals from remaining work and operation count.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Quick deadline check
    if not _within_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each operation, compute per-machine slack and a composite urgency score
    scores = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            # Fallback: use remaining work
            scores[i] = remaining_work[i]
            continue

        durations = np.array(list(machine_options.values()), dtype=np.float64)
        min_dur = np.min(durations)
        max_dur = np.max(durations)
        mean_dur = np.mean(durations)
        n_machines = len(durations)

        # Slack for this operation on its best machine (least duration)
        # Slack = remaining_work - duration on best machine (proxy for idle time created)
        best_slack = remaining_work[i] - min_dur

        # Soft-min over all machine slacks (log-sum-exp based)
        # slack_j = remaining_work[i] - duration_j
        slacks = remaining_work[i] - durations
        # Soft-min: -logsumexp(-slacks / T) * T for temperature T
        T = 1.0
        neg_slacks = -slacks / T
        # Numerically stable logsumexp
        max_neg = np.max(neg_slacks)
        if max_neg > 1e6:
            max_neg = 1e6
        logsumexp_val = max_neg + np.log(np.sum(np.exp(neg_slacks - max_neg)))
        soft_min_slack = -logsumexp_val * T

        # Urgency: operations with more remaining work and more remaining ops are more critical
        # Use a combined urgency measure
        # WSPT-like: remaining_work / remaining_ops (work per remaining op)
        if remaining_ops[i] > 0:
            wspt = remaining_work[i] / remaining_ops[i]
        else:
            wspt = remaining_work[i]

        # SPT-like: just remaining_work
        spt = remaining_work[i]

        # Machine flexibility: more machines = more options = less bottleneck risk
        flexibility = np.log1p(n_machines)

        # Duration variability: high variability means the choice of machine matters more
        if n_machines > 1:
            variability = (max_dur - min_dur) / (mean_dur + 1e-9)
        else:
            variability = 0.0

        # Composite score:
        # 1. Soft-min slack (higher is better - more buffer)
        # 2. Hard WSPT urgency (higher is more urgent)
        # 3. SPT urgency
        # 4. Flexibility bonus (more machines = safer)
        # 5. Variability penalty (high variability on tight slack = risky)
        
        # Normalize components roughly
        score = (
            0.30 * soft_min_slack +
            0.25 * wspt +
            0.20 * spt +
            0.15 * flexibility * 10.0 +  # scale flexibility
            -0.10 * variability * min(1.0, abs(best_slack) / (remaining_work[i] + 1e-9))
        )
        scores[i] = score

    # Deadline check before final selection
    if not _within_deadline():
        return int(np.argmax(remaining_work))

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
