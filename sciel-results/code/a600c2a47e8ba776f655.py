"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global start time for deadline guard
_START_TIME = _time.monotonic()
_DEADLINE = 140.0  # Safety margin before 150s kill

# Module-level state for machine load tracking
_machine_load = {}
# Track number of operations completed per machine (for average duration estimation)
_machine_op_count = {}
# Track total duration assigned per machine
_machine_total_duration = {}


def _pick_machine(op, time):
    """
    Choose machine balancing processing time with congestion avoidance.
    Uses a smooth surrogate: minimize duration + congestion_penalty * current_load
    """
    machine_options = op["machine_options"]
    if not machine_options:
        return None
    
    best_machine = None
    best_score = float('inf')
    
    for machine, duration in machine_options.items():
        current_load = _machine_load.get(machine, 0.0)
        # Penalty for assigning to a busy machine
        # Use a smooth penalty: load / (load + duration) as congestion indicator
        congestion = current_load / (current_load + duration + 1e-9)
        
        # Score: prefer less congested machines, shorter durations
        # We want to MINIMIZE this score
        machine_score = duration + 0.15 * current_load + 0.5 * congestion * duration
        
        if machine_score < best_score:
            best_score = machine_score
            best_machine = machine
    
    return best_machine


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Uses a dynamic bottleneck optimization by prioritizing the operation that 
    maximizes the ratio of its job's remaining critical path length to the 
    estimated cumulative load of its eligible machines.
    """
    global _machine_load
    
    if not ready_operations:
        return 0

    # Deadline guard: if we are running low on time, fall back to a very fast heuristic
    if _time.monotonic() - _START_TIME > _DEADLINE:
        # Fallback: Most work remaining first (fast)
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        idx = 0
        m = _pick_machine(ready_operations[0], time)
        if m is not None:
            # Update load
            dur = ready_operations[0]["machine_options"][m]
            _machine_load[m] = _machine_load.get(m, 0.0) + dur
            _machine_op_count[m] = _machine_op_count.get(m, 0) + 1
            _machine_total_duration[m] = _machine_total_duration.get(m, 0.0) + dur
            return (idx, m)
        return idx

    # Gather features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    spt = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Compute estimated machine load for each operation's best machine
    # This is the "dynamic bottleneck" component
    est_loads = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        m = _pick_machine(op, time)
        if m is not None:
            best_machines[i] = m
            # Use the current tracked load as an estimate of future delay
            est_loads[i] = _machine_load.get(m, 0.0)
        else:
            # Fallback to shortest processing time machine
            if op["machine_options"]:
                m = min(op["machine_options"].keys(), key=lambda x: op["machine_options"][x])
                best_machines[i] = m
                est_loads[i] = _machine_load.get(m, 0.0)

    def _normalize(a):
        """Normalize to [0, 1] range; higher is better for the objective."""
        rng = a.max() - a.min()
        if rng < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - a.min()) / rng

    norm_work = _normalize(remaining_works)
    norm_ops = _normalize(remaining_ops)
    
    # Urgency ratio: High remaining work relative to short processing time
    ratio = remaining_works / (spt + 1e-9)
    norm_ratio = _normalize(ratio)
    
    # Dynamic bottleneck metric: 
    # We want to prioritize operations where the job has significant work left 
    # but the estimated machine delay is low relative to that work.
    # Metric: remaining_work / (est_load + spt + epsilon)
    # Higher value means the job has lots of work but the machine is relatively free/fast.
    bottleneck_ratio = remaining_works / (est_loads + spt + 1e-9)
    norm_bottleneck = _normalize(bottleneck_ratio)

    # Weights: 
    # - Bottleneck (0.4): Primary driver, balances job urgency with machine availability
    # - Work (0.3): Secondary driver for makespan reduction (MWRF)
    # - Ratio (0.2): Captures urgency/density
    # - Ops (0.1): Tie-breaker to keep jobs with many steps in the pipeline
    W_BOTTLENECK = 0.4
    W_WORK = 0.3
    W_RATIO = 0.2
    W_OPS = 0.1

    scores = (
        W_BOTTLENECK * norm_bottleneck +
        W_WORK * norm_work +
        W_RATIO * norm_ratio +
        W_OPS * norm_ops
    )

    idx = int(np.argmax(scores))

    # Choose machine: use the pre-computed best machine for this operation
    m = best_machines[idx]
    
    if m is not None:
        # Update machine load for the chosen machine
        duration = ready_operations[idx]["machine_options"].get(m, 0.0)
        _machine_load[m] = _machine_load.get(m, 0.0) + duration
        _machine_op_count[m] = _machine_op_count.get(m, 0) + 1
        _machine_total_duration[m] = _machine_total_duration.get(m, 0.0) + duration
        return (idx, m)
    else:
        # Fallback if no machine was found (should not happen)
        return idx

# EVOLVE-BLOCK-END
