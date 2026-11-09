"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = _time.time() + 145.0  # 5s safety margin
_DEADLINE_CHECK_INTERVAL = 0.05   # seconds between checks

# Track machine loads across dispatch calls for load-balancing heuristic
_machine_loads = {}  # machine -> cumulative load (approximate)
_last_time = 0.0


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def _soft_min(values, temperature=1.0):
    """
    Soft-min via log-sum-exp: -temperature * log(sum(exp(-v/temperature)))
    This is a smooth surrogate for min(values) that captures the full distribution.
    As temperature -> 0, it approaches the true min.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    # Numerically stable log-sum-exp
    shifted = np.asarray(values, dtype=np.float64) / temperature
    max_shifted = np.max(shifted)
    if not np.isfinite(max_shifted):
        return min(values)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    return -temperature * log_sum_exp


def _soft_max(values, temperature=1.0):
    """
    Soft-max via log-sum-exp: temperature * log(sum(exp(v/temperature)))
    Smooth surrogate for max(values). As temperature -> 0, approaches true max.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    shifted = np.asarray(values, dtype=np.float64) / temperature
    max_shifted = np.max(shifted)
    if not np.isfinite(max_shifted):
        return max(values)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    return temperature * log_sum_exp


def _update_machine_loads(ready_operations, time, chosen_index):
    """
    Update machine load estimates based on the chosen operation.
    We assign the operation to the machine with minimum (load + duration),
    which is a greedy load-balancing heuristic.
    """
    global _last_time
    if chosen_index < 0 or chosen_index >= len(ready_operations):
        return
    
    op = ready_operations[chosen_index]
    machine_options = op["machine_options"]
    
    if not machine_options:
        return
    
    # Find the machine with minimum (current_load + duration)
    best_machine = None
    best_cost = float('inf')
    for m, dur in machine_options.items():
        current_load = _machine_loads.get(m, 0.0)
        cost = current_load + dur
        if cost < best_cost:
            best_cost = cost
            best_machine = m
    
    if best_machine is not None:
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + machine_options[best_machine]
    
    _last_time = time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Composite scoring that blends:
    1. Remaining work (job completion priority)
    2. Soft-min of machine durations (smooth surrogate for min duration, captures flexibility)
    3. Number of remaining ops (job urgency)
    4. Work density (remaining_work / remaining_ops) - identifies bottleneck jobs
    5. Machine load penalty (soft-max of load+duration across machines) - load balancing
    
    The score is a weighted composite. We pick the operation with the highest score.
    """
    global _last_time
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft-min of machine durations: smooth surrogate for min, captures distribution
    soft_min_durs = np.array([
        _soft_min(list(op["machine_options"].values()), temperature=0.5)
        for op in ready_operations
    ], dtype=np.float64)

    # Work density: remaining_work / max(remaining_ops, 1)
    safe_ops = np.maximum(remaining_ops, 1.0)
    work_density = remaining_works / safe_ops

    # Machine load penalty: for each op, compute soft-max over (load_m + dur_m)
    # Higher penalty means the op would overload a busy machine -> prefer lower penalty
    load_penalties = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            load_penalties[i] = 0.0
            continue
        costs = []
        for m, dur in machine_options.items():
            current_load = _machine_loads.get(m, 0.0)
            costs.append(current_load + dur)
        # Use soft-max to capture the worst-case machine assignment
        load_penalties[i] = _soft_max(costs, temperature=0.3)

    # Normalize each feature to [0, 1] range
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    norm_work = _normalize(remaining_works)
    norm_soft_dur = _normalize(soft_min_durs)
    norm_ops = _normalize(remaining_ops)
    norm_density = _normalize(work_density)
    norm_load = _normalize(load_penalties)

    # Weights: prioritize remaining work, then soft-min duration (lower is better -> invert),
    # then work density (higher is better), then fewer remaining ops (lower is better -> invert),
    # then lower load penalty (lower is better -> invert)
    w1 = 0.35  # remaining work
    w2 = 0.20  # soft-min duration (lower is better, so we invert)
    w3 = 0.15  # work density (higher is better)
    w4 = 0.10  # remaining ops (lower is better, so we invert)
    w5 = 0.20  # load penalty (lower is better, so we invert)

    # For duration, remaining ops, and load penalty, lower is better, so we invert
    score = (w1 * norm_work + 
             w2 * (1.0 - norm_soft_dur) + 
             w3 * norm_density + 
             w4 * (1.0 - norm_ops) +
             w5 * (1.0 - norm_load))

    # Deadline guard after computation
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    chosen = int(np.argmax(score))
    
    # Update machine load estimates
    _update_machine_loads(ready_operations, time, chosen)
    
    return chosen
# EVOLVE-BLOCK-END
