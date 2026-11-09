"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s; keep a safety margin.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin


def _time_remaining():
    return _DEADLINE - _time.time()


def dispatch(ready_operations, time):
    """
    Dynamic bottleneck-aware dispatch rule.
    
    Strategy: For each ready operation, identify its best machine (shortest duration).
    Then score each operation by a combination of:
      1. WSPT urgency (remaining_work / min_duration) - captures job urgency
      2. Machine load balancing - prefer operations that go to least-loaded machines
      3. Slack consideration - avoid starving machines by looking at remaining ops
    
    We select the operation that maximizes a composite score designed to
    minimize makespan by balancing urgency with machine utilization.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: simple WSPT
        scores = []
        for op in ready_operations:
            min_dur = min(op["machine_options"].values())
            scores.append(op["remaining_work"] / min_dur)
        return int(np.argmax(scores))

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])

    # Avoid division by zero
    min_durations_safe = np.where(min_durations > 0, min_durations, 1.0)

    # WSPT ratio: remaining_work / min_duration
    wspt = remaining_works / min_durations_safe

    # For each operation, find the best machine (shortest duration)
    best_machine_durs = min_durations.copy()
    best_machines = []
    for op in ready_operations:
        mo = op["machine_options"]
        best_m = min(mo.keys(), key=lambda m: mo[m])
        best_machines.append(best_m)

    # Compute machine load: for each machine, sum of durations of ready ops that would use it
    # This is a proxy for how busy each machine will be
    machine_load = {}
    for i, op in enumerate(ready_operations):
        best_m = best_machines[i]
        best_d = best_machine_durs[i]
        machine_load[best_m] = machine_load.get(best_m, 0.0) + best_d

    # Normalize machine loads
    total_load = sum(machine_load.values()) if machine_load else 1.0
    machine_load_norm = {}
    for m, load in machine_load.items():
        machine_load_norm[m] = load / total_load if total_load > 0 else 0.0

    # Score: combine WSPT with machine load balancing
    # Prefer high WSPT (urgent) and low machine load (less congested)
    scores = np.zeros(n)
    for i in range(n):
        best_m = best_machines[i]
        m_load = machine_load_norm.get(best_m, 0.0)
        # Higher WSPT is better; lower machine load is better
        # Use a weighted combination
        scores[i] = wspt[i] * (1.0 + 0.5 * (1.0 - m_load))

    # Also consider: operations with fewer remaining ops might be "critical path"
    # Add a small bonus for operations that are last in their job (to avoid idle)
    # Actually, operations with MORE remaining ops after this one are more urgent
    # because delaying them delays the whole job
    # remaining_ops likely means ops remaining AFTER this one
    # So higher remaining_ops = more work queued behind = more urgent
    # This is already captured in remaining_work

    # Final selection
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
