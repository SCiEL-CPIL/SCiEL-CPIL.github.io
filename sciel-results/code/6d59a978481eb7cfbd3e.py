"""Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 115.0  # safety margin before 120s kill


def _check_deadline():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Fast path: if we're close to deadline, just pick first
    if _check_deadline():
        return 0

    # Compute per-operation features
    scores = []
    for op in ready_operations:
        mo = op["machine_options"]
        machines = list(mo.keys())
        durations = [mo[m] for m in machines]
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / len(durations)
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        # Machine load proxy: prefer machines with fewer ops / shorter durations
        # Use min duration as best-case completion, and penalize spread
        # Critical ratio: remaining_work / (time + 1) as urgency
        # We want to minimize makespan, so:
        #   - Pick ops that finish quickly on some machine (min_dur low)
        #   - Pick ops that are critical to their job (high remaining_work, few remaining_ops)
        #   - Avoid long ops early unless they're on a dedicated machine

        # Score components (lower is better for selection)
        # 1. Duration penalty: shorter is better
        dur_score = min_dur + 0.3 * (max_dur - min_dur)  # penalize variability

        # 2. Job criticality: more remaining work on fewer ops = more urgent
        if remaining_ops > 0:
            work_per_op = remaining_work / remaining_ops
        else:
            work_per_op = remaining_work

        # 3. Urgency: jobs with less remaining time should go first
        #    (remaining_work is total remaining, so high = more work left = less urgent per unit time)
        #    Actually for makespan, we want to keep machines busy; 
        #    prioritize ops that unlock more work or are on bottleneck machines.

        # Composite: favor short, high-criticality ops
        # Normalize-ish: use remaining_ops as a weight for criticality
        criticality = work_per_op * (1.0 + remaining_ops * 0.1)

        # Final score: lower = better
        # Weight duration heavily since it directly affects makespan
        score = dur_score * 1.0 + criticality * 0.15
        scores.append(score)

    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    return best_idx
# EVOLVE-BLOCK-END
