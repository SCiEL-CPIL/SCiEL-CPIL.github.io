"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.time() > _DEADLINE


def _soft_min(values, temperature=0.1):
    """Approximate min using log-sum-exp with negative values (soft-min)."""
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    v_min = v.min()
    exp_vals = np.exp(-(v - v_min) / temperature)
    return v_min + temperature * np.log(exp_vals.sum())


# Global state to track cumulative load per machine for load-balancing
_machine_load = {}


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path for single op
    if n == 1:
        return 0

    # Extract key metrics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each operation, find the least-loaded eligible machine and its projected completion time
    # Also compute soft-min duration and weighted-SPT (remaining_work / best_duration)
    soft_min_durs = np.empty(n, dtype=np.float64)
    best_durs = np.empty(n, dtype=np.float64)
    projected_completions = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        durs = np.array(list(op["machine_options"].values()), dtype=np.float64)
        machines = list(op["machine_options"].keys())
        soft_min_durs[i] = _soft_min(durs, temperature=0.1)
        best_durs[i] = durs.min()

        # Find the machine with the least projected completion time
        # Projected completion = current time + machine load + operation duration
        min_completion = float('inf')
        for m in machines:
            load = _machine_load.get(m, 0.0)
            dur = op["machine_options"][m]
            completion = time + load + dur
            if completion < min_completion:
                min_completion = completion

        projected_completions[i] = min_completion

    # Normalization helper
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Urgency: more remaining work and more remaining ops => more urgent
    urgency_work = _normalize(remaining_works)
    urgency_ops = _normalize(remaining_ops)
    urgency = 0.7 * urgency_work + 0.3 * urgency_ops

    # Load-balancing score: earlier projected completion => higher score
    load_balance = _normalize(-projected_completions)

    # Weighted SPT: remaining_work / best_duration (critical path per unit time)
    # Avoid division by zero
    safe_best_durs = np.where(best_durs > 1e-12, best_durs, 1.0)
    w_spt = remaining_works / safe_best_durs
    w_spt_norm = _normalize(w_spt)

    # Combined score: urgency + load balancing + weighted SPT
    scores = 1.0 * urgency + 0.5 * load_balance + 0.3 * w_spt_norm

    # Pick the operation with the highest combined score
    best_idx = int(np.argmax(scores))

    # Update global machine load for the selected operation
    # Choose the least-loaded eligible machine for this operation
    op = ready_operations[best_idx]
    machines = list(op["machine_options"].keys())
    min_completion = float('inf')
    best_machine = None
    for m in machines:
        load = _machine_load.get(m, 0.0)
        dur = op["machine_options"][m]
        completion = time + load + dur
        if completion < min_completion:
            min_completion = completion
            best_machine = m

    # Update the load for the chosen machine
    if best_machine is not None:
        dur = op["machine_options"][best_machine]
        _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + dur

    return best_idx
# EVOLVE-BLOCK-END
