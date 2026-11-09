import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = 140.0  # safety margin before 150s kill

# Persistent machine free-time tracker (when each machine becomes available)
_machine_free_time = {}


def dispatch(ready_operations, time):
    """Dispatch rule: EDF with machine-load-aware deadlines, SPT, and critical-path urgency."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _time.monotonic() - _START > _DEADLINE - 5.0:
        # Fallback: pick min duration (SPT)
        durs = [min(op.get("machine_options", {0: 1}).values(), default=1) for op in ready_operations]
        return int(np.argmin(durs))

    scores = np.zeros(n, dtype=np.float64)
    min_durs = np.zeros(n, dtype=np.float64)
    rem_work = np.zeros(n, dtype=np.float64)
    rem_ops = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    best_durations = np.zeros(n, dtype=np.float64)
    virtual_finish = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {0: 1})
        durs = list(mo.values())
        d_min = min(durs) if durs else 1.0
        min_durs[i] = d_min

        rw = op.get("remaining_work", 0)
        rem_work[i] = rw if rw > 0 else 1.0

        ro = op.get("remaining_ops", 0)
        rem_ops[i] = ro if ro > 0 else 1.0

        # Find the best machine (min duration) and its current free time
        best_machine = min(mo, key=mo.get)
        best_machines[i] = best_machine
        best_durations[i] = mo[best_machine]

        m_free = _machine_free_time.get(best_machine, time)
        # Virtual finish time: when this op would finish if scheduled on its best machine now
        virtual_finish[i] = m_free + best_durations[i]

    # Normalize features
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng

    norm_vf = _normalize(virtual_finish)          # EDF: earlier finish is better
    norm_spt = _normalize(1.0 / (min_durs + 1e-9))  # SPT: shorter is better
    norm_rw = _normalize(rem_work)                # critical path: more remaining work = more urgent
    norm_ratio = _normalize(rem_work / (min_durs + 1e-9))  # urgency ratio

    # Composite score (lower is better for selection)
    # EDF is primary: prioritize operations that finish soonest on their best machine
    # SPT secondary: reduce blocking
    # Critical path tertiary: don't starve jobs with lots of remaining work
    w_edf = 0.45
    w_spt = 0.30
    w_rw = 0.15
    w_ratio = 0.10

    scores = (w_edf * norm_vf +
              w_spt * (1.0 - norm_spt) +  # invert: shorter processing = lower score
              w_rw * (1.0 - norm_rw) +    # invert: more remaining work = lower score (more urgent)
              w_ratio * (1.0 - norm_ratio))

    idx = int(np.argmin(scores))

    # Update machine free time for the selected operation
    selected_machine = best_machines[idx]
    selected_duration = best_durations[idx]
    current_free = _machine_free_time.get(selected_machine, time)
    _machine_free_time[selected_machine] = current_free + selected_duration

    return idx
