"""Open-Shop Scheduling dispatch rule with load-aware heuristics."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Module-level deadline guard to ensure we finish well within limits
_START_TIME = _time.time()
_DEADLINE = 115.0  # seconds; safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Load-aware dispatch that balances:
    1. Machine congestion (prefer machines with less current load)
    2. Job urgency (jobs with more remaining ops/work are more critical)
    3. Shortest processing time on the least-loaded feasible machine
    4. Machine flexibility (more options = more scheduling freedom)
    
    Returns: (index, machine) tuple to let the simulator assign the operation
    to its fastest machine, directly reducing makespan.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        if ready_operations:
            mo = ready_operations[0].get("machine_options", {})
            if mo:
                best_m = min(mo, key=mo.get)
                return (0, best_m)
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        mo = ready_operations[0].get("machine_options", {})
        if mo:
            best_m = min(mo, key=mo.get)
            return (0, best_m)
        return 0

    # For each ready operation, compute a score
    # We want to minimize makespan, so we prefer:
    # - Operations that can go on less-loaded machines (to balance load)
    # - Operations from jobs with more remaining work (urgency)
    # - Shorter durations on the best available machine
    # - Operations with more machine options (flexibility)

    scores = np.zeros(n)
    best_machines = [None] * n

    # Pre-compute urgency factors
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # Normalize urgency to [0, 1]
    max_rw = remaining_works.max() if remaining_works.max() > 0 else 1.0
    max_ro = remaining_ops.max() if remaining_ops.max() > 0 else 1.0

    # Track min duration across all ops for normalization
    min_durs = np.zeros(n)
    flexibilities = np.zeros(n)
    soft_min_durs = np.zeros(n)  # Smooth surrogate for min duration

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            scores[i] = 0.0
            continue

        durations = np.array(list(mo.values()), dtype=float)
        min_dur = float(np.min(durations))
        min_durs[i] = min_dur
        flexibilities[i] = len(durations)

        # Soft-min: smooth approximation of min using log-sum-exp
        # soft_min = -log(sum(exp(-x_i / tau)) * tau) for small tau
        tau = 0.1
        if len(durations) > 1:
            neg_durs = -durations / tau
            max_neg = np.max(neg_durs)
            log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_durs - max_neg)))
            soft_min_val = -tau * log_sum_exp
            soft_min_durs[i] = soft_min_val
        else:
            soft_min_durs[i] = min_dur

        # Find the machine with the shortest duration (hard min for actual assignment)
        best_m = min(mo, key=mo.get)
        best_machines[i] = best_m

    # Normalize durations and flexibility
    max_min_dur = min_durs.max() if min_durs.max() > 0 else 1.0
    min_min_dur = min_durs.min() if min_durs.min() < max_min_dur else 0.0
    range_min_dur = max_min_dur - min_min_dur if max_min_dur > min_min_dur else 1.0
    max_flex = flexibilities.max() if flexibilities.max() > 0 else 1.0

    for i in range(n):
        mo = ready_operations[i]["machine_options"]
        if not mo:
            continue

        # Urgency: higher remaining work/ops means this job is more critical
        urgency = (remaining_works[i] / max_rw) * 0.4 + (remaining_ops[i] / max_ro) * 0.6

        # Duration factor: shorter is better, using soft-min for smoothness
        # Normalize soft_min_dur to [0, 1] where 0 = shortest (best)
        dur_factor = 1.0 - ((soft_min_durs[i] - min_min_dur) / range_min_dur)

        # Flexibility factor: more machine options is better
        flex_factor = flexibilities[i] / max_flex

        # Penalty for low flexibility: operations with only 1 option are less flexible
        # This is already captured in flex_factor, but we add a small penalty
        if flexibilities[i] == 1:
            flex_penalty = 0.05
        else:
            flex_penalty = 0.0

        # Combine: duration is most important for makespan, then urgency, then flexibility
        # Weighted sum with duration dominating (shorter jobs reduce makespan directly)
        scores[i] = dur_factor * 0.45 + urgency * 0.40 + flex_factor * 0.15 - flex_penalty

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))

    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        if best_machines[best_idx] is not None:
            return (best_idx, best_machines[best_idx])
        return best_idx

    # Return (index, machine) to let simulator assign to fastest machine
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    return best_idx
# EVOLVE-BLOCK-END
