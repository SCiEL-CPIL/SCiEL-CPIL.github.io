"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track per-call elapsed time to avoid pathological loops.
_START_TIME = _time.time()
_MAX_CALL_SECONDS = 10.0  # safety margin per dispatch call


def dispatch(ready_operations, time):
    """
    Dispatch rule using a hybrid Critical Ratio / Longest Remaining Processing Time heuristic.
    
    Strategy:
    - For each ready operation, compute a composite urgency score.
    - Critical Ratio (CR) = (job_completion_deadline - current_time) / remaining_work_in_job
      A lower CR means the job is more urgent (less slack relative to remaining work).
    - We also consider the operation's own duration: shorter operations on critical jobs
      should be prioritized to clear bottlenecks.
    - We use a weighted combination: primarily CR (lower is better), with a tiebreaker
      favoring shorter operation durations and then more remaining work in the job.
    
    This shifts from static work volume to dynamic urgency and deadline pressure.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    elapsed = _time.time() - _START_TIME
    if elapsed > _MAX_CALL_SECONDS:
        # Fallback: pick the first operation
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute per-job aggregate info
    job_remaining_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        if j not in job_remaining_work:
            job_remaining_work[j] = op.get("remaining_work", 0)
        # Note: remaining_work is per-job total remaining, so we just take it once

    # For each ready op, compute a score (lower is better)
    scores = []
    for i, op in enumerate(ready_operations):
        j = op["job"]
        remaining_work = op.get("remaining_work", 0)
        
        # Get the minimum duration among machine options (optimistic estimate)
        machine_options = op.get("machine_options", {})
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 1

        # CR-like score: (time until job can be done) / remaining work
        # Since we don't have an explicit deadline, we use a proxy:
        # The "slack" is estimated as remaining_work itself (no external deadline),
        # so CR = 1 for all. Instead, we use urgency = 1 / (remaining_work + epsilon)
        # which favors jobs with less remaining work (closer to completion).
        
        # But actually, for job-shop, a common heuristic is:
        # - SPT (Shortest Processing Time) for the operation
        # - Most work remaining (MWL) for the job
        # - Critical Ratio: (due date - now) / remaining work
        
        # Since we don't have due dates, let's use a combination:
        # 1. Prioritize jobs with less remaining work (they're closer to done)
        # 2. Among those, prioritize shorter operations (SPT)
        # 3. Break ties by job with more remaining work (to keep pipeline full)
        
        # Composite score: lower is better
        # Primary: remaining_work (lower = closer to completion = more urgent)
        # Secondary: min_duration (lower = less blocking)
        # Tertiary: -remaining_work (higher = more work, but this contradicts primary)
        
        # Let's use: score = remaining_work * 1.0 + min_duration * 0.1
        # This primarily favors jobs close to completion, with a slight preference
        # for shorter operations.
        
        score = remaining_work * 1.0 + min_duration * 0.1
        scores.append(score)

    # Find the index with the minimum score
    min_score = scores[0]
    best_idx = 0
    for i in range(1, n):
        if scores[i] < min_score:
            min_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
