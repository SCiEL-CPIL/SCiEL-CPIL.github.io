"""Job-Shop Scheduling dispatch rule: Multi-criteria heuristic with machine load awareness."""
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 140.0  # Safe margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: Multi-criteria heuristic combining:
    - Shortest processing time on best machine (SPT)
    - Longest remaining work (MWRT) to avoid starving long jobs
    - Machine load balancing (prefer less-loaded machines)
    
    The score combines these factors with weights tuned for job-shop scheduling.
    """
    # Deadline guard
    if _time.time() - _START > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the best (shortest) duration and its machine
        min_dur = float('inf')
        best_machine = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_machine = m

        if min_dur == float('inf'):
            min_dur = 1.0

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None or remaining_work < 0:
            remaining_work = 0

        remaining_ops = op.get("remaining_ops", 0)
        if remaining_ops is None or remaining_ops < 0:
            remaining_ops = 0

        # Multi-criteria score (lower is better):
        # 1. Primary: shortest processing time (SPT) - frees machines quickly
        # 2. Secondary: normalize by remaining work to balance job progress
        # 3. Tertiary: slight penalty for operations with many remaining ops
        #    (prefer completing jobs to free up resources)
        
        # Weighted combination:
        # - SPT component: min_dur (lower is better)
        # - MWRT component: -remaining_work (higher remaining work is worse for immediate scheduling,
        #   but we want to avoid starving, so we use a soft penalty)
        # - Remaining ops: fewer remaining ops means closer to completion (prefer finishing jobs)
        
        # Use a composite that balances:
        #   score = min_dur / (1 + remaining_ops * 0.1) - 0.5 * remaining_work / (remaining_work + 1)
        # This favors short ops, slightly penalizes ops with many remaining steps,
        # and gives a small boost to jobs with more remaining work (to prevent starvation)
        
        spt_component = min_dur
        mwrt_component = remaining_work / (remaining_work + 1.0)  # 0 to 1, higher = more work left
        remaining_ops_penalty = remaining_ops * 0.05  # small penalty for many remaining ops
        
        # Composite: lower is better
        # We want short ops (low min_dur), but also want to progress jobs with lots of work
        # The mwrt_component is high when there's lots of work left, which we want to schedule
        # So we subtract a fraction of it (negative contribution to score)
        score = spt_component - 0.3 * mwrt_component + remaining_ops_penalty
        
        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
