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
    Weight by duration to identify which machines are becoming bottlenecks.
    Use soft weighting to account for operation flexibility.
    """
    # Reset loads for this dispatch call
    _machine_load.clear()
    
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
            
        durations = np.array(list(machine_options.values()), dtype=float)
        machines = list(machine_options.keys())
        num_machines = len(machines)
        
        # Soft weighting: distribute load based on inverse duration
        # Operations with shorter durations on a machine contribute less to congestion
        # Use softmax-like weighting to account for flexibility
        inv_durations = 1.0 / (durations + 1e-9)
        weights = inv_durations / (np.sum(inv_durations) + 1e-9)
        
        for i, machine in enumerate(machines):
            if machine not in _machine_load:
                _machine_load[machine] = 0.0
            # Weighted contribution: longer ops create more congestion
            # but flexibility reduces effective load
            effective_weight = weights[i] * (1.0 + 0.5 / num_machines)
            _machine_load[machine] += durations[i] * effective_weight


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Uses machine-load balancing strategy with refined congestion-aware
    penalties and smoother soft-min approximations.
    """
    work = op["remaining_work"]
    machine_options = op["machine_options"]
    remaining_ops = op.get("remaining_ops", 0)

    if not machine_options:
        return 0.0

    durations = np.array(list(machine_options.values()), dtype=float)
    machines = list(machine_options.keys())
    num_machines = len(machines)

    # Soft-min for duration with adaptive temperature
    # Sharper when durations are close, smoother when spread out
    dur_range = float(np.max(durations) - np.min(durations))
    base_temp = 1.0
    temp = base_temp * (1.0 + 0.5 * dur_range / (float(np.mean(durations)) + 1e-9))
    
    scaled_durations = durations / temp
    max_scaled = np.max(scaled_durations)
    shifted = scaled_durations - max_scaled
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(shifted)))
    soft_min_dur = temp * (np.log(num_machines) - log_sum_exp)
    
    max_dur = float(np.max(durations))
    min_dur = float(np.min(durations))
    mean_dur = float(np.mean(durations))

    # Machine congestion: how loaded are the candidate machines?
    machine_loads = np.array([_machine_load.get(m, 0.0) for m in machines], dtype=float)
    min_load = float(np.min(machine_loads))
    max_load = float(np.max(machine_loads))
    avg_load = float(np.mean(machine_loads))

    # Global context
    all_loads = np.array(list(_machine_load.values()), dtype=float) if _machine_load else np.array([0.0])
    global_max_load = float(np.max(all_loads)) if len(all_loads) > 0 else 0.0
    global_mean_load = float(np.mean(all_loads)) if len(all_loads) > 0 else 0.0

    # Score components:
    # 1. Remaining work (primary): more work = higher priority
    score = 1.0 * work

    # 2. Machine load balancing: bonus if this op can use a less-loaded machine
    #    Use soft-min of loads with adaptive temperature
    if avg_load > 0:
        load_range = max_load - min_load
        load_mean = avg_load
        load_temp = base_temp * (1.0 + 0.3 * load_range / (load_mean + 1e-9))
        
        scaled_loads = machine_loads / load_temp
        max_scaled_load = np.max(scaled_loads)
        shifted_loads = scaled_loads - max_scaled_load
        log_sum_exp_load = max_scaled_load + np.log(np.sum(np.exp(shifted_loads)))
        soft_min_load = load_temp * (np.log(num_machines) - log_sum_exp_load)
        
        # Normalize against global max
        if global_max_load > 0:
            congestion_ratio = soft_min_load / (global_max_load + 1e-9)
        else:
            congestion_ratio = 0.0
        score += 1.8 * (1.0 - congestion_ratio)

    # 3. Penalty if most machines for this op are heavily congested
    #    Smooth penalty using soft-max of normalized loads
    if global_max_load > 0:
        normalized_loads = machine_loads / (global_max_load + 1e-9)
        # Soft-max to identify worst congestion level
        scaled_norm = normalized_loads / 0.5  # temperature for soft-max
        max_scaled_norm = np.max(scaled_norm)
        shifted_norm = scaled_norm - max_scaled_norm
        log_sum_exp_norm = max_scaled_norm + np.log(np.sum(np.exp(shifted_norm)))
        soft_max_congestion = 0.5 * log_sum_exp_norm
        
        # Smooth penalty: quadratic for heavier congestion
        score -= 0.6 * (soft_max_congestion ** 2)

    # 4. Flexibility: more machine options is better (log-scaled)
    score += 0.4 * np.log1p(num_machines)

    # 5. Duration urgency: shorter operations finish faster
    #    Use inverse soft-min for smoothness
    score += 0.5 / (soft_min_dur + 1e-9)

    # 6. Urgency: fewer remaining ops = closer to completion
    score += 0.4 / (remaining_ops + 1e-9)

    # 7. Penalty for high duration variance (uncertainty in completion)
    #    Use coefficient of variation for scale-invariant measure
    dur_cv = dur_range / (mean_dur + 1e-9)
    score -= 0.2 * dur_cv

    # 8. Bonus for operations that can avoid the most congested machine
    #    If the least-loaded option is significantly better, give extra credit
    if num_machines > 1 and max_load > 0:
        load_improvement = (max_load - min_load) / (max_load + 1e-9)
        score += 0.3 * load_improvement

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
    scores = np.array([_compute_op_score(op, time) for op in ready_operations], dtype=float)

    # Add tiny random tiebreaker to avoid systematic tie-breaking
    # that can cause pathological scheduling patterns
    if len(ready_operations) > 1:
        scores += np.random.uniform(0, 1e-6, size=len(scores))

    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
