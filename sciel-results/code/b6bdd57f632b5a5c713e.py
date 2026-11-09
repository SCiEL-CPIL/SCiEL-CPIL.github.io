"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard: track elapsed wall-clock time.
_START_TIME = _time.perf_counter()
_DEADLINE = 150.0 - 5.0  # 5s safety margin before 150s kill


def _elapsed_ok():
    return (_time.perf_counter() - _START_TIME) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule combining:
    1. Dynamic bottleneck optimization: prioritize operations that maximize
       the ratio of remaining critical path work to estimated machine congestion.
    2. Shortest processing time on the chosen machine to reduce idle time.
    3. Work-to-time ratio to prioritize jobs with dense remaining work.
    
    Returns (index, machine) to explicitly select both the operation and machine.
    """
    if not ready_operations:
        return 0
    
    if not _elapsed_ok():
        # Safety fallback: pick operation with most remaining work
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return best_idx

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            best_machine = min(machine_options, key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0

    best_score = float('-inf')
    best_index = 0
    best_machine = None

    # Pre-compute minimum duration for each operation (for congestion estimation)
    min_durations = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            min_durations.append(min(mo.values()))
        else:
            min_durations.append(float('inf'))

    # Estimate total cumulative load (sum of min durations) across all ready ops
    # This represents the minimum possible total work if all ops used their best machine
    total_min_load = sum(d for d in min_durations if d < float('inf'))
    if total_min_load <= 0:
        total_min_load = 1.0

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 1)
        if ro <= 0:
            ro = 1

        # Work-to-time ratio: how much work per remaining op
        work_ratio = rw / ro

        # For each eligible machine, compute a composite score
        for machine, duration in machine_options.items():
            # Component 1: Dynamic bottleneck ratio
            # Prioritize operations whose remaining work is high relative to
            # the estimated congestion (total min load of all ready ops)
            # This avoids scheduling ops that would add to already-congested machines
            bottleneck_ratio = rw / (total_min_load + 1.0)

            # Component 2: Shortest processing time (lower duration is better)
            # Use negative duration so shorter is preferred
            spf_component = -duration

            # Component 3: Work-to-time ratio (higher is better)
            # Jobs with more dense work should be prioritized
            wtt_component = work_ratio

            # Component 4: Remaining ops bonus to prevent starvation of long jobs
            ro_bonus = 0.1 * ro

            # Composite score: weighted combination
            # Weights tuned to balance congestion avoidance, speed, and work density
            score = (
                0.35 * bottleneck_ratio +   # dynamic bottleneck: avoid congestion
                0.30 * (spf_component / max(1.0, duration)) +  # normalized SPT
                0.25 * (wtt_component / max(1.0, rw)) +  # normalized work-to-time
                ro_bonus
            )

            # Small tiebreaker: prefer lower machine indices to reduce contention
            if isinstance(machine, (int, float)):
                score -= 0.001 * machine

            if score > best_score:
                best_score = score
                best_index = i
                best_machine = machine

    if best_machine is not None:
        return (best_index, best_machine)
    else:
        return best_index
# EVOLVE-BLOCK-END
