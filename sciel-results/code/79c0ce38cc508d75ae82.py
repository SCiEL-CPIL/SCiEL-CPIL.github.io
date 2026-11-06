"""Rewritten flexible job-shop dispatch with a composite urgency/load-aware heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


# Global deadline guard (120 s hard limit; keep a safety margin)
_DEADLINE_START = _time.monotonic()
_HARD_LIMIT = 115.0  # seconds; leave 5 s margin before 120 s kill
_DEADLINE = _DEADLINE_START + _HARD_LIMIT


def _elapsed_ok():
    return _time.monotonic() < _DEADLINE


def dispatch(ready_operations, time):
    """Pick which ready operation to start next and, optionally, which machine.

    Returns an integer index (preferred) or (index, machine) tuple.
    The heuristic combines:
      - Job urgency (remaining work / remaining ops) to prioritize jobs with large remaining effort
      - Immediate processing speed (min duration on eligible machines) to reduce local makespan
      - Machine load balancing via a soft penalty based on current time and expected load
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if not _elapsed_ok():
        return 0

    scores = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        # Extract key features
        remaining_work = float(op.get("remaining_work", 0.0))
        remaining_ops = max(1, int(op.get("remaining_ops", 1)))
        machine_options = op.get("machine_options", {})

        if not machine_options:
            scores[i] = 0.0
            continue

        # Minimum duration among eligible machines (fastest option)
        min_dur = min(machine_options.values())
        if min_dur <= 0:
            min_dur = 1e-9

        # Job urgency: remaining work per remaining op (average remaining op size)
        urgency = remaining_work / remaining_ops

        # Speed factor: prefer operations that can be done quickly on some machine
        speed = 1.0 / min_dur

        # Composite score: higher is better
        # Weight urgency heavily, then speed
        score = urgency * 0.7 + speed * 0.3

        # Soft load-balancing penalty: penalize if the fastest machine is likely busy
        # We don't have direct machine load info, but we can use the current time
        # as a proxy: later in the schedule, prefer operations with more flexibility
        # (more machine options) to avoid bottlenecks
        flexibility = len(machine_options)
        score += flexibility * 0.05

        scores[i] = score

    # Return the index of the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
