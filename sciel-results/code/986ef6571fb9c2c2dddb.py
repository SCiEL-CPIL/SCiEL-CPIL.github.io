"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


_START_TIME = _time.time()
_DEADLINE = 120.0  # seconds; evaluator kills at 120s, use 150s wall-clock budget but be safe


def _ensure_deadline():
    """Return True if we are past the safety deadline."""
    return (_time.time() - _START_TIME) > (_DEADLINE - 2.0)


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Heuristic components tuned for open-shop makespan minimization:
    1. Remaining work: jobs with more total remaining work are more critical.
    2. Effective duration: use the minimum duration (best-case machine) to estimate
       how quickly this op can finish and unlock successors.
    3. Remaining ops: fewer remaining ops means the job is closer to completion
       (urgency / finishing soon).
    4. Flexibility penalty: if all machines have long durations, the op is a bottleneck.
    """
    work = op.get("remaining_work", 0)
    machine_options = op.get("machine_options", {})
    
    if not machine_options:
        return 0.0
    
    durations = list(machine_options.values())
    min_dur = min(durations)
    max_dur = max(durations)
    
    remaining_ops = op.get("remaining_ops", 0)
    
    # Soft-min proxy: use min_dur as the best-case completion time.
    # For a single machine option, min_dur == max_dur.
    
    # Primary: remaining work – the job with the most work left is most critical
    # for makespan because it contributes the most to the critical path.
    score = 1.0 * work
    
    # Secondary: prefer operations that can finish quickly (short min duration)
    # so that subsequent operations in the job can start sooner.
    score += 0.5 / (min_dur + 1e-9)
    
    # Tertiary: prefer jobs with fewer remaining ops (closer to done).
    # Completing a job frees its machine slots for other jobs.
    score += 0.3 / (remaining_ops + 1e-9)
    
    # Penalty for high duration spread: if the best machine is much shorter than
    # the worst, there is flexibility; if all are long, this op is a bottleneck.
    # Use a soft penalty based on the ratio max/min.
    if min_dur > 0:
        spread_ratio = max_dur / (min_dur + 1e-9)
        # Penalize when spread is low (all machines slow) – less flexibility
        score -= 0.1 * max(0.0, 1.0 - spread_ratio)
    
    # Additional: prefer operations where the best machine duration is a small
    # fraction of remaining work (i.e., the op is not the dominant remaining piece)
    if work > 0:
        dur_fraction = min_dur / (work + 1e-9)
        # If the op duration is a large fraction of remaining work, it's critical
        score += 0.2 * dur_fraction
    
    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Deadline guard: if we're running out of time, just pick the first
    # operation with the most remaining work (fast fallback).
    if _ensure_deadline():
        # Fast fallback: pick op with max remaining_work
        best_idx = 0
        best_work = ready_operations[0].get("remaining_work", 0)
        for i, op in enumerate(ready_operations[1:], 1):
            w = op.get("remaining_work", 0)
            if w > best_work:
                best_work = w
                best_idx = i
        return best_idx
    
    # Compute scores for all ready operations
    scores = [_compute_op_score(op, time) for op in ready_operations]
    
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
