"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Track machine availability (estimated completion time) for lookahead
_mach_avail = {}
# Track total load assigned to each machine for load balancing
_mach_load = {}


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: simple urgency-based dispatch
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick machine that minimizes projected completion + load penalty
            best_machine = min(
                machine_options,
                key=lambda m: max(_mach_avail.get(m, 0.0), time) + machine_options[m] + 0.1 * _mach_load.get(m, 0.0)
            )
            _mach_avail[best_machine] = max(_mach_avail.get(best_machine, 0.0), time) + machine_options[best_machine]
            _mach_load[best_machine] = _mach_load.get(best_machine, 0.0) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Number of machine options (flexibility)
    num_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    
    # Min duration for speed
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Earliest completion time estimate
    # Use min available time among eligible machines + min duration
    earliest_completions = np.array([
        max(time, min(_mach_avail.get(m, 0.0) for m in op["machine_options"])) + min(op["machine_options"].values())
        if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Normalize features to [0, 1]
    # 1. Urgency: Average remaining work per op (higher is more critical)
    avg_work_per_op = remaining_works / (remaining_ops + 1.0)
    max_avg_work = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    norm_urgency = avg_work_per_op / (max_avg_work + 1e-9)

    # 2. Speed: Inverse of min duration (higher is faster)
    speed_scores = 1.0 / (min_durations + 1e-9)
    max_speed = np.max(speed_scores) if np.max(speed_scores) > 0 else 1.0
    norm_speed = speed_scores / (max_speed + 1e-9)

    # 3. Constraint: Inverse of number of options (higher is more constrained)
    constraint_scores = 1.0 / (num_options + 1e-9)
    max_constraint = np.max(constraint_scores) if np.max(constraint_scores) > 0 else 1.0
    norm_constraint = constraint_scores / (max_constraint + 1e-9)

    # 4. Earliest Completion: Inverse of completion time (higher is sooner)
    # Filter out inf
    valid_earliest = earliest_completions[earliest_completions < np.inf]
    if len(valid_earliest) > 0:
        min_earliest = np.min(valid_earliest)
        max_earliest = np.max(valid_earliest)
        range_earliest = max_earliest - min_earliest if max_earliest > min_earliest else 1.0
        # Invert: earlier is better (higher score)
        norm_earliest = 1.0 - (earliest_completions - min_earliest) / (range_earliest + 1e-9)
        norm_earliest = np.clip(norm_earliest, 0.0, 1.0)
        norm_earliest[earliest_completions == np.inf] = 0.0
    else:
        norm_earliest = np.zeros(n)

    # Weighted scoring
    # Based on Inspiration 1, speed/efficiency is important.
    # Constraint helps avoid bottlenecks on single-machine ops.
    w_urgency = 1.0
    w_speed = 0.8
    w_constraint = 0.6
    w_earliest = 0.4

    scores = (w_urgency * norm_urgency +
              w_speed * norm_speed +
              w_constraint * norm_constraint +
              w_earliest * norm_earliest)

    best_idx = int(np.argmax(scores))

    # Choose the machine that minimizes projected completion + load penalty
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(
            machine_options,
            key=lambda m: max(_mach_avail.get(m, 0.0), time) + machine_options[m] + 0.1 * _mach_load.get(m, 0.0)
        )
        # Update availability and load
        _mach_avail[best_machine] = max(_mach_avail.get(best_machine, 0.0), time) + machine_options[best_machine]
        _mach_load[best_machine] = _mach_load.get(best_machine, 0.0) + machine_options[best_machine]
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
