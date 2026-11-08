"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE_MARGIN = 2.0  # seconds to stay under the 150s wall-clock limit
_START_TIME = None


def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME


def _deadline_exceeded():
    return _now() > (150.0 - _DEADLINE_MARGIN)


def _min_duration(op):
    """Minimum processing time among available machines for this operation."""
    opts = op.get("machine_options", {})
    if not opts:
        return 0
    return min(opts.values())


def _critical_path_weight(op):
    """
    Estimate the remaining critical path length for this job.
    Uses remaining_work as a proxy, but weights by number of remaining ops
    to capture urgency of finishing the job.
    """
    remaining_ops = op.get("remaining_ops", 0)
    remaining_work = op.get("remaining_work", 0)
    if remaining_ops <= 0:
        return remaining_work
    # Average work per remaining op, scaled by ops count to favor jobs
    # that are close to completion (fewer ops left = more urgent)
    return remaining_work + 0.5 * remaining_work / max(remaining_ops, 1)


def _wspt_ratio(op):
    """Weighted Shortest Processing Time: min_duration / remaining_work."""
    remaining_work = op.get("remaining_work", 0)
    if remaining_work <= 0:
        return 0.0
    return _min_duration(op) / remaining_work


def _urgency_bonus(op):
    """Bonus for jobs with fewer remaining operations (closer to done)."""
    remaining_ops = op.get("remaining_ops", 0)
    if remaining_ops <= 0:
        return 1.0
    return 1.0 / remaining_ops


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a multi-criteria heuristic:
    - Critical path weight (primary): favor operations that contribute most to
      the remaining makespan of their job.
    - WSPT ratio (secondary): favor shorter processing times relative to remaining work.
    - Urgency bonus (tertiary): favor jobs closer to completion.
    
    Returns the index of the selected operation.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    if _deadline_exceeded():
        # Fallback: pick the one with most remaining work (original heuristic)
        return int(max(range(len(ready_operations)),
                       key=lambda i: ready_operations[i].get("remaining_work", 0)))

    n = len(ready_operations)
    scores = []

    for i in range(n):
        op = ready_operations[i]
        cpw = _critical_path_weight(op)
        wspt = _wspt_ratio(op)
        urgency = _urgency_bonus(op)

        # Combine: critical path dominates, then WSPT, then urgency
        # Normalize-ish: cpw is in same units as time, wspt is ratio, urgency is small
        score = cpw * 1.0 + wspt * 0.3 + urgency * 0.1
        scores.append(score)

    # Pick the operation with the highest combined score
    best_idx = max(range(n), key=lambda i: scores[i])
    return int(best_idx)
# EVOLVE-BLOCK-END
