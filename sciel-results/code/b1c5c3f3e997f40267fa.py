"""Job-Shop Scheduling Dispatch Rule.
Refinement: Ratio-based priority with soft-min machine selection and
critical path adjustment.
Score = soft_min_duration / (1 + remaining_work) + critical_bonus.
Uses log-sum-exp for smooth machine selection instead of hard min,
and adds a small bonus for operations on the critical path (high remaining_ops).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 120s/150s limit


def _soft_min(values, temperature=0.5):
    """Compute soft-min via log-sum-exp for smooth selection."""
    if len(values) == 1:
        return float(values[0])
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    if np.isinf(v_min):
        return float("inf")
    exp_vals = np.exp(-(v - v_min) / temperature)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return float(v_min)
    return float(v_min + temperature * np.log(sum_exp))


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Deadline guard: fall back to pure SPT (min min-duration) with best machine
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = np.empty(n, dtype=np.float64)
        best_m = [None] * n
        for i, op in enumerate(ready_operations):
            mo = op.get("machine_options", {})
            if mo:
                bm, bd = min(mo.items(), key=lambda kv: kv[1])
                min_durs[i] = bd
                best_m[i] = bm
            else:
                min_durs[i] = float("inf")
        idx = int(np.argmin(min_durs))
        if best_m[idx] is not None:
            return (idx, best_m[idx])
        return idx

    # ---- Extract features and compute refined ratio-based scores ----
    scores = np.empty(n, dtype=np.float64)
    best_mach = [None] * n

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            durations = list(mo.values())
            machines = list(mo.keys())
            # Use soft-min for smooth machine selection
            soft_min_dur = _soft_min(durations, temperature=0.3)
            # Also track the hard-min machine for return value
            bm = min(machines, key=lambda m: mo[m])
            best_mach[i] = bm
            min_dur = soft_min_dur
        else:
            min_dur = float("inf")
            best_mach[i] = None

        rw = float(op.get("remaining_work", 0.0))
        remaining_ops = int(op.get("remaining_ops", 1))

        # Refined ratio-based score with critical path adjustment
        # Base: min_dur / (1 + rw) — prefer short ops on jobs with lots of remaining work
        # Critical bonus: small penalty for jobs with many remaining operations
        # (encourages making progress on long chains to avoid bottlenecks)
        if np.isinf(min_dur):
            scores[i] = float("inf")
        else:
            base_score = min_dur / (1.0 + rw)
            # Critical path bonus: jobs with more remaining ops get a small
            # negative adjustment (lower score = higher priority)
            critical_bonus = -0.05 * remaining_ops
            scores[i] = base_score + critical_bonus

    # Select operation with minimum score
    best_idx = int(np.argmin(scores))

    bm = best_mach[best_idx]
    if bm is not None:
        return (best_idx, bm)
    return best_idx

# EVOLVE-BLOCK-END
