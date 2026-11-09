"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def _smooth_min(values, temperature=1.0):
    """Smooth min via log-sum-exp: approximates min(v_i) smoothly."""
    if len(values) == 0:
        return 0.0
    v = np.array(values, dtype=np.float64)
    if len(v) == 1:
        return v[0]
    v_min = np.min(v)
    exp_terms = np.exp(-(v - v_min) / temperature)
    log_sum = np.log(np.sum(exp_terms))
    return v_min + temperature * log_sum


def _z_norm(x):
    """Z-score normalization; returns zeros if std is negligible."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def dispatch(ready_operations, time):
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        if op["machine_options"]:
            best_machine = min(op["machine_options"], key=lambda m: op["machine_options"][m])
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Smooth min duration (soft-min via log-sum-exp)
    smooth_min_durations = np.array([
        _smooth_min(list(op["machine_options"].values()), temperature=1.0) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Exact min duration for machine selection
    exact_min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Flexibility: number of machine options
    n_machines = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Intensity: remaining work per remaining op
    safe_ops = np.maximum(remaining_ops, 1.0)
    intensity = remaining_works / safe_ops

    # Z-score normalize each feature
    z_intensity = _z_norm(intensity)
    z_ops = _z_norm(remaining_ops)
    z_min_dur = _z_norm(smooth_min_durations)
    z_flex = _z_norm(n_machines)

    # Time-aware urgency: as time increases, prioritize shorter durations
    # Estimate total horizon from max remaining work
    max_remaining = np.max(remaining_works) if len(remaining_works) > 0 else 1.0
    time_frac = min(time / max(max_remaining, 1e-6), 1.0)
    # Urgency weight grows with time fraction
    urgency_weight = 0.2 + 0.3 * time_frac

    # Composite score (maximize)
    w_intensity = 1.0
    w_ops = 0.4
    w_min_dur = -0.3 - urgency_weight  # stronger SPT pressure later
    w_flex = 0.1

    score = (w_intensity * z_intensity +
             w_ops * z_ops +
             w_min_dur * z_min_dur +
             w_flex * z_flex)

    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)

    best_idx = int(np.argmax(score))

    # Select best machine for the chosen operation (shortest duration)
    op = ready_operations[best_idx]
    if op["machine_options"]:
        best_machine = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
