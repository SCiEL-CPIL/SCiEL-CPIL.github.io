"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE_START = None
_DEADLINE_LIMIT = 145.0  # 5s safety margin before 150s kill


def _check_deadline():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.time()
    return (_time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For SPT: use the minimum duration across machine options for the current op
    spt_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Normalize each criterion to [0, 1] range
    def normalize(arr, reverse=False):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        normed = (arr - mn) / (mx - mn)
        if reverse:
            normed = 1.0 - normed
        return normed

    # MWR: higher remaining work = more urgent (higher score)
    mwr_norm = normalize(remaining_works, reverse=False)

    # SPT: shorter processing time = better (lower duration is better, so reverse)
    spt_norm = normalize(spt_durations, reverse=True)

    # Fewer remaining ops: fewer ops left = less complexity, but we want to prioritize
    # jobs with more remaining work. Actually, fewer remaining ops means the job is
    # closer to completion, which could be good to finish. Let's use: fewer ops = higher score
    fewer_ops_norm = normalize(remaining_ops, reverse=True)

    # Weighted combination: MWR 0.5, SPT 0.3, fewer-ops 0.2
    scores = 0.5 * mwr_norm + 0.3 * spt_norm + 0.2 * fewer_ops_norm

    # Look-ahead: simulate completing each ready operation and estimate the
    # reduction in the global makespan lower bound (max remaining work).
    # The operation that most reduces the critical path gets a bonus.
    max_remaining = remaining_works.max()
    
    # For each candidate, estimate how much the max remaining work would drop
    # if we completed this operation now.
    # After completing op i, its remaining_work drops by spt_durations[i].
    # The new max is max(remaining_works with op i reduced, other ops unchanged)
    lookahead_scores = np.zeros(n)
    for i in range(n):
        new_works = remaining_works.copy()
        new_works[i] = max(0.0, remaining_works[i] - spt_durations[i])
        new_max = new_works.max()
        # Reduction in makespan lower bound
        lookahead_scores[i] = max_remaining - new_max

    # Normalize lookahead scores
    la_norm = normalize(lookahead_scores, reverse=False)

    # Combine with original scores
    # Weight: original 0.7, lookahead 0.3
    final_scores = 0.7 * scores + 0.3 * la_norm

    # Check deadline
    if _check_deadline():
        return int(np.argmax(scores))

    return int(np.argmax(final_scores))
# EVOLVE-BLOCK-END
