"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Track elapsed time across dispatch calls for a global deadline guard.
_START_TIME = _time.time()
# Safety margin: stop heavy work well before the 150s wall-clock kill.
_DEADLINE_SECONDS = 140.0

# Per-machine load estimate (cumulative assigned work). We approximate machine
# occupancy using the durations we observe being dispatched. This is a heuristic
# proxy for congestion; it is not a full simulation but biases us away from
# machines that have already received a lot of work.
_machine_load = {}


def _elapsed():
    return _time.time() - _START_TIME


def _reset_if_stale():
    """Reset per-instance state if we appear to be in a fresh instance.

    We cannot know instance boundaries explicitly, so we reset the load map
    conservatively: if the elapsed time suggests a new evaluation window is
    unlikely mid-instance we keep it. To be safe and cheap, we simply cap the
    growth and rely on the fact that each instance is short. We do NOT reset
    automatically to avoid losing useful load info within an instance.
    """
    pass


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.

    Returns an int index into ready_operations (the operation to schedule).
    Optionally could return (index, machine); we return int to match the seed
    interface exactly.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Deadline guard: if we are dangerously close to the kill, fall back to a
    # cheap deterministic rule immediately.
    if _elapsed() > _DEADLINE_SECONDS:
        # Cheap fallback: most remaining work.
        best = 0
        best_rw = ready_operations[0].get("remaining_work", 0)
        for i in range(1, n):
            rw = ready_operations[i].get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best = i
        return int(best)

    # Precompute per-operation features.
    scores = np.empty(n, dtype=np.float64)

    # For machine-load bias, we need to know which machine each op would use.
    # We choose the machine with the smallest (duration + current_load) for each op,
    # then use that combined value as a congestion proxy.
    for i in range(n):
        op = ready_operations[i]
        mo = op.get("machine_options", {})
        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)

        if not mo:
            # No machine options: treat as low priority.
            scores[i] = -1e9
            continue

        # Pick the machine that minimizes (duration + load).
        best_val = None
        for m, dur in mo.items():
            load = _machine_load.get(m, 0.0)
            val = dur + load
            if best_val is None or val < best_val:
                best_val = val

        # Feature 1: job criticality (more remaining work => higher priority).
        f_work = remaining_work

        # Feature 2: urgency (fewer remaining ops => tighter; add a small bonus).
        # Use inverse-ish: fewer remaining ops is more urgent.
        f_urgency = (1.0 / (1.0 + remaining_ops)) if remaining_ops is not None else 0.0

        # Feature 3: congestion (lower is better). Normalize-ish by subtracting.
        f_congestion = -best_val if best_val is not None else 0.0

        # Weighted combination. Tuned weights: work is primary, congestion
        # secondary, urgency tertiary.
        score = 1.0 * f_work + 0.5 * f_urgency + 0.3 * f_congestion
        scores[i] = score

        # Update the chosen machine's load estimate with the chosen duration so
        # that subsequent dispatches in the same instance see increased load.
        # We need the chosen machine; recompute quickly (cheap, small mo).
        best_m = None
        best_v = None
        for m, dur in mo.items():
            load = _machine_load.get(m, 0.0)
            val = dur + load
            if best_v is None or val < best_v:
                best_v = val
                best_m = m
        if best_m is not None:
            _machine_load[best_m] = _machine_load.get(best_m, 0.0) + (best_v - _machine_load.get(best_m, 0.0))
            # Simpler: just add the duration.
            # (Overwrite with a clean add.)
        # Clean add of duration:
        if best_m is not None:
            _machine_load[best_m] = _machine_load.get(best_m, 0.0) + mo[best_m]

    # Pick the argmax.
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
