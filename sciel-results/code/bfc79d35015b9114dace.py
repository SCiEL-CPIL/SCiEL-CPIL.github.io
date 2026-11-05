"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = _time.time()

# Track machine loads to identify bottlenecks
_machine_load = {}


def _ensure_deadline():
    """Return True if we are within the wall-clock budget."""
    return (_time.time() - _START_TIME) < _DEADLINE


def _update_machine_loads(ready_operations, time):
    """
    Update machine load estimates based on ready operations.
    This helps identify which machines are becoming bottlenecks.
    """
    # Reset loads for this dispatch call
    _machine_load.clear()
    
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in _machine_load:
                _machine_load[machine] = 0.0
            _machine_load[machine] += duration


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Uses machine-load balancing strategy: prioritize operations on the most
    constrained machines (highest load or fewest options) to prevent bottlenecks.
    """
    work = op["remaining_work"]
    machine_options = op["machine_options"]
    remaining_ops = op.get("remaining_ops", 0)

    if not machine_options:
        return 0.0

    durations = list(machine_options.values())
    min_dur = min(durations)
    max_dur = max(durations)
    avg_dur = sum(durations) / len(durations)
    num_machines = len(machine_options)

    # Machine constraint score: prefer operations that can use less-loaded machines
    # or machines with more options (more flexibility)
    machine_loads = [_machine_load.get(m, 0.0) for m in machine_options.keys()]
    min_load = min(machine_loads) if machine_loads else 0.0
    max_load = max(machine_loads) if machine_loads else 0.0
    avg_load = sum(machine_loads) / len(machine_loads) if machine_loads else 0.0

    # Score components:
    # 1. Remaining work (primary): more work = higher priority
    # 2. Machine load balancing: prefer ops that can run on less-loaded machines
    # 3. Flexibility: prefer ops with more machine options
    # 4. Duration: shorter operations finish faster
    # 5. Urgency: fewer remaining ops = closer to completion

    score = 0.0
    score += 1.0 * work  # Most work remaining
    
    # Machine load balancing: bonus if this op can use a less-loaded machine
    if avg_load > 0:
        score += 0.5 * (1.0 - min_load / (avg_load + 1e-9))  # Higher if min_load is low relative to avg
    
    # Flexibility: more machine options is better
    score += 0.2 * np.log1p(num_machines)
    
    # Shorter operations are preferred
    score += 0.3 / (min_dur + 1e-9)
    
    # Urgency: fewer remaining ops
    score += 0.2 / (remaining_ops + 1e-9)

    # Penalty for high variance in durations (uncertainty)
    if max_dur > min_dur:
        score -= 0.1 * (max_dur - min_dur) / (min_dur + 1e-9)

    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast rule
    if not _ensure_deadline():
        # Fast fallback: pick the operation with the most remaining work
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations[1:], 1):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        return best_idx

    # Update machine load estimates
    _update_machine_loads(ready_operations, time)

    # Compute scores for all ready operations
    scores = [_compute_op_score(op, time) for op in ready_operations]

    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
