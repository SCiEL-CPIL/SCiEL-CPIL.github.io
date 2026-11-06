"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; keep a safety margin.
_DEADLINE = 145.0
_START_TIME = _time.time()


def _time_left():
    return _DEADLINE - _time.time()


def _simulate(ready_ops_snapshot, time_snapshot):
    """
    Greedy simulation using a composite heuristic:
    - Pick operation with highest 'urgency' = remaining_work / remaining_ops
      (average remaining work per remaining op), tie-broken by total remaining_work.
    - Assign to the machine that becomes free earliest (min current load + duration).
    Returns the makespan (completion time of all operations).
    """
    if not ready_ops_snapshot:
        return time_snapshot

    # Build a mutable copy of the scheduling state
    # We need to simulate forward: at each step, pick one ready op, assign to a machine,
    # advance time, and potentially make other ops ready.

    # Since we only have the snapshot of ready operations at the current time,
    # and we don't have the full problem instance (job dependencies, etc.),
    # we must work with what's given. The dispatch function is called repeatedly
    # by the simulator, so we just need to pick the best one from the ready set.

    # However, for a better heuristic, let's look at the structure:
    # Each op has: job, op, machine_options {machine: duration}, remaining_ops, remaining_work

    # A good open-shop heuristic:
    # 1. Among ready ops, pick the one with the largest remaining_work (most work left in that job)
    #    OR the one whose job has the most remaining ops (to keep jobs progressing)
    # 2. Assign it to the machine with the least total assigned work so far (or least current load)

    # Since we can't track machine loads across calls (stateless dispatch),
    # we use a local heuristic based on the ready set itself.

    # Heuristic: score = remaining_work * (1 + 0.1 * remaining_ops)
    # This prioritizes operations from jobs with lots of remaining work and ops.

    # Better: use a weighted combination
    # score = alpha * remaining_work + beta * (remaining_work / max(1, remaining_ops))
    # The second term is average remaining work per op, favoring jobs with heavy remaining ops.

    best_idx = 0
    best_score = -1e18

    for i, op in enumerate(ready_ops_snapshot):
        rw = op["remaining_work"]
        ro = max(1, op["remaining_ops"])
        avg_rw = rw / ro

        # Composite score: prioritize high total remaining work and high average remaining work
        score = rw + 0.5 * avg_rw

        # Tie-break: prefer operations with fewer machine options (more constrained)
        n_machines = len(op["machine_options"])
        score += 0.1 * (5 - n_machines)  # smaller n_machines -> higher bonus

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    Returns an index into ready_operations.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time_left() < 0.1:
        # Emergency fallback: pick first
        return 0

    # Use the composite heuristic
    return _simulate(ready_operations, time)
# EVOLVE-BLOCK-END
