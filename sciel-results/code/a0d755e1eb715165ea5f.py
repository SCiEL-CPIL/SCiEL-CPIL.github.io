"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 110.0  # safety margin before 120s kill


def _elapsed():
    return _time.monotonic() - _DEADLINE_START


def _pick_machine(op, time):
    """
    Pick the machine that minimizes the expected completion time of this operation,
    considering both the duration and a heuristic for machine load.
    """
    machine_options = op.get("machine_options", {})
    if not machine_options:
        return 0

    best_machine = None
    best_score = float('inf')

    for machine, duration in machine_options.items():
        # Heuristic: prefer shorter duration, but also consider remaining work
        # We don't have direct access to machine load in this interface,
        # so we use duration as the primary factor.
        # A more sophisticated approach could weight by remaining_ops.
        score = duration
        if score < best_score:
            best_score = score
            best_machine = machine

    return best_machine if best_machine is not None else 0


def _score_operation(op, time):
    """
    Score a ready operation. Lower score = better (should be dispatched sooner).
    We want to prioritize operations that:
    1. Have the most remaining work (to avoid leaving long jobs for last)
    2. Have fewer remaining operations (to reduce chain length)
    3. Can be completed quickly on their best machine
    """
    remaining_work = op.get("remaining_work", 0)
    remaining_ops = op.get("remaining_ops", 0)
    machine_options = op.get("machine_options", {})

    # Find the minimum duration across available machines
    if machine_options:
        min_duration = min(machine_options.values())
    else:
        min_duration = 0

    # Composite score:
    # - High remaining work should be prioritized (so we invert: negative contribution)
    # - Fewer remaining ops is better (lower is better)
    # - Shorter processing time is better (lower is better)
    #
    # We use a weighted sum. The idea is that operations with lots of remaining work
    # should be started early to avoid them becoming bottlenecks.
    #
    # Weight tuning:
    # - remaining_work has the largest impact on makespan
    # - min_duration affects immediate completion
    # - remaining_ops affects the chain of dependencies

    if remaining_ops <= 0:
        remaining_ops = 1

    score = -remaining_work * 1.0 + min_duration * 0.5 + remaining_ops * 0.1
    return score


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Returns the index of the ready operation to start next.
    Optionally returns (index, machine) tuple.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _elapsed() > _DEADLINE_LIMIT:
        # Fall back to simple rule to avoid timeout
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    # Compute scores for each ready operation
    scores = [_score_operation(op, time) for op in ready_operations]

    # Pick the operation with the lowest score (best)
    best_idx = int(np.argmin(scores))

    # Pick the best machine for that operation
    best_machine = _pick_machine(ready_operations[best_idx], time)

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
