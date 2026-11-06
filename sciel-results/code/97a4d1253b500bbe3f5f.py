"""Job-Shop Scheduling dispatch rule with composite heuristic."""
# EVOLVE-BLOCK-START
import time as _time

_start_time = None


def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()

    # Deadline guard: if we're getting close to the 120s limit, return a safe default
    elapsed = _time.time() - _start_time
    if elapsed > 110:
        # Return first ready operation as safe fallback
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Compute composite score for each ready operation
    # We want to maximize this score (pick the operation with highest score)
    scores = []

    # Pre-compute some statistics
    max_remaining_work = 0
    max_remaining_ops = 0
    max_duration = 0

    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        # Get the minimum duration among machine options (best case)
        durations = list(op.get("machine_options", {}).values())
        if durations:
            dur = min(durations)
        else:
            dur = 0
        if rw > max_remaining_work:
            max_remaining_work = rw
        if ro > max_remaining_ops:
            max_remaining_ops = ro
        if dur > max_duration:
            max_duration = dur

    # Avoid division by zero
    if max_remaining_work == 0:
        max_remaining_work = 1
    if max_remaining_ops == 0:
        max_remaining_ops = 1
    if max_duration == 0:
        max_duration = 1

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        durations = list(op.get("machine_options", {}).values())
        if durations:
            dur = min(durations)
        else:
            dur = 0

        # Component 1: Most Work Remaining (normalized)
        mwr_score = rw / max_remaining_work

        # Component 2: Most Remaining Operations (normalized)
        mro_score = ro / max_remaining_ops

        # Component 3: Longest Processing Time (normalized)
        lpt_score = dur / max_duration

        # Component 4: "Urgency" - jobs with fewer remaining ops but significant work
        # are close to completion and should be prioritized to free resources
        urgency_score = 0
        if ro > 0:
            # Higher urgency for jobs that are close to finishing
            urgency_score = (max_remaining_ops - ro) / max_remaining_ops

        # Component 5: Work per remaining operation (density)
        density_score = 0
        if ro > 0:
            density_score = rw / (ro * max_remaining_work)
            # Normalize
            density_score = min(density_score, 1.0)

        # Composite score with weights tuned for job-shop scheduling
        # Weight LPT heavily (bottleneck machines), then MWR, then urgency
        score = (
            0.35 * lpt_score +
            0.30 * mwr_score +
            0.15 * mro_score +
            0.10 * urgency_score +
            0.10 * density_score
        )

        scores.append(score)

    # Return index of the operation with the highest composite score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
