"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# --- Deadline guard ---------------------------------------------------------
# The evaluator kills the process at 120 s wall-clock. dispatch() is invoked
# once per scheduling event. Per-call work is tiny (<= 10 ops x 5 machines),
# but we keep an explicit guard so that if (unexpectedly) we approach the
# limit we degrade to a cheap O(1) greedy choice instead of risking a timeout.
_START = _time.time()
_DEADLINE = _START + 115.0  # 5 s safety margin before the 120 s kill


def _min_duration(op):
    """Minimum processing time across eligible machines (fastest option)."""
    opts = op.get("machine_options", {})
    best = float("inf")
    best_m = None
    for m, d in opts.items():
        if d < best:
            best = d
            best_m = m
    if best_m is None:
        # No options: treat as zero duration, unknown machine.
        return 0.0, None
    return best, best_m


def _critical_path(op):
    """
    Estimate the job's remaining critical path length = sum, over every
    remaining operation, of its minimum processing time. This is the
    Longest-Remaining-Processing-Time (LRPT) / critical-path priority used in
    top FJSP dispatch rules.
    """
    rem_ops = op.get("remaining_ops")
    if not rem_ops:
        # Fall back to this operation's own min duration.
        d, _ = _min_duration(op)
        return d
    total = 0.0
    for r in rem_ops:
        if isinstance(r, dict):
            d, _ = _min_duration(r)
            total += d
        elif isinstance(r, (int, float)):
            # If remaining_ops is a list of durations.
            total += float(r)
        else:
            # Unknown structure; ignore.
            pass
    if total == 0.0:
        d, _ = _min_duration(op)
        total = d
    return total


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.

    Returns (index, machine) so the simulator explicitly starts the chosen
    ready operation on its fastest eligible machine. If the interface expects
    a bare index, returning a tuple is the documented optional form.

    Priority (dominant -> minor):
      1. Critical path (LRPT): job with the largest remaining critical path
         length is the makespan bottleneck -> schedule it first.
      2. Urgency density: remaining_work / min_duration (progress per unit time).
      3. Fewer remaining operations (closer to completion).
      4. Smallest min duration (SPT) to free the machine quickly.

    Machine: the eligible machine with the minimum duration (earliest finish).
    """
    if not ready_operations:
        return 0

    # --- Deadline guard: cheap fallback near the limit ---------------------
    if _time.time() > _DEADLINE:
        # O(1)-ish: pick the operation with the smallest min duration.
        bi = 0
        bd = float("inf")
        for i, op in enumerate(ready_operations):
            d, _ = _min_duration(op)
            if d < bd:
                bd = d
                bi = i
        return int(bi)

    best_index = 0
    best_key = None
    best_machine = None

    for i, op in enumerate(ready_operations):
        min_dur, machine = _min_duration(op)
        crit = _critical_path(op)
        rem_work = float(op.get("remaining_work", 0.0) or 0.0)
        rem_ops = op.get("remaining_ops")
        n_rem = len(rem_ops) if isinstance(rem_ops, (list, tuple, set)) else 1

        # Urgency density: how much work we knock out per unit of this step.
        if min_dur > 0:
            density = rem_work / min_dur
        else:
            density = float("inf") if rem_work > 0 else 0.0

        # Composite key, lexicographic (larger = better):
        #   primary  : critical path length (bottleneck first)
        #   secondary: urgency density
        #   tertiary : -remaining ops (finish soon)
        #   quaternary: -min duration (SPT)
        key = (
            crit,
            density,
            -n_rem,
            -min_dur,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_index = i
            best_machine = machine

    # Return the explicit (index, machine) form when we have a machine,
    # otherwise fall back to a bare index.
    if best_machine is not None:
        return (int(best_index), best_machine)
    return int(best_index)
# EVOLVE-BLOCK-END
