"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0


def _check_deadline(fallback):
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def _soft_min(values, temperature=0.8):
    """Soft min via log-sum-exp: approximates min(v_i) smoothly."""
    if len(values) == 0:
        return 0.0
    v = np.array(values, dtype=np.float64)
    if len(v) == 1:
        return v[0]
    v_min = np.min(v)
    if v_min > 1e15:
        return v_min
    exp_terms = np.exp(-(v - v_min) / temperature)
    log_sum = np.log(np.sum(exp_terms))
    return v_min + temperature * log_sum


def _z_norm(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


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

    # --- Extract features ---
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft min and hard min of durations per operation
    soft_min_durs = np.array([
        _soft_min(list(op["machine_options"].values()), temperature=0.8) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    hard_min_durs = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    n_machines = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    safe_ops = np.maximum(remaining_ops, 1.0)
    intensity = remaining_works / safe_ops

    # Time-normalized urgency factor
    max_rw = np.max(remaining_works) if np.any(remaining_works > 0) else 1.0
    t_norm = min(time / max(max_rw, 1e-6), 3.0)

    # --- Slack / criticality computation ---
    # For each op, slack = remaining_work - current_time (how much "pressure")
    # An op with high remaining work and few remaining ops is more critical.
    # Use a soft-max of (intensity * remaining_ops) to capture joint criticality.
    criticality_raw = intensity * safe_ops  # = remaining_work (redundant but clear)
    # Better: criticality = remaining_work / (1 + remaining_ops * 0.3)
    # Ops with many remaining ops are spread out; few ops means concentrated pressure.
    criticality = remaining_works / (1.0 + 0.3 * remaining_ops)

    # Time-aware: later in schedule, criticality matters more
    w_crit = 1.0 + 0.4 * t_norm
    w_intensity = 0.8
    w_ops = 0.4
    w_dur = -0.5 - 0.15 * t_norm  # stronger SPT pressure later
    w_flex = 0.12
    w_slack = 0.25  # slack-based term

    z_crit = _z_norm(criticality)
    z_intensity = _z_norm(intensity)
    z_ops = _z_norm(remaining_ops)
    z_dur = _z_norm(soft_min_durs)
    z_flex = _z_norm(n_machines)

    # Slack term: prefer ops whose soft-min duration is well below their intensity
    # (i.e., they can finish quickly relative to their workload pressure)
    slack_proxy = criticality - np.maximum(soft_min_durs, 0.0)
    z_slack = _z_norm(slack_proxy)

    score = (w_crit * z_crit + w_intensity * z_intensity + w_ops * z_ops +
             w_dur * z_dur + w_flex * z_flex + w_slack * z_slack)

    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)

    # --- Pair-level scoring: for the top candidates, evaluate (op, machine) pairs ---
    # Instead of picking one op then one machine, score pairs jointly.
    # We'll evaluate all (i, m) pairs where i is in ready_operations and m in op_i's machine_options.
    # Score pair = op_score[i] - machine_cost(i, m)
    # machine_cost captures: duration on that machine + opportunity cost (how many other ops
    # would prefer this machine, weighted by their durations).

    # Precompute: for each machine, the sum of durations of other ops that could use it
    # (opportunity cost proxy)
    all_machines = set()
    for op in ready_operations:
        if op["machine_options"]:
            all_machines.update(op["machine_options"].keys())

    # For each machine, compute the "demand" = sum of min-durations of ops that could use it
    machine_demand = {}
    for m in all_machines:
        total = 0.0
        for op in ready_operations:
            if op["machine_options"] and m in op["machine_options"]:
                total += op["machine_options"][m]
        machine_demand[m] = total

    # Normalize machine demand
    if machine_demand:
        max_demand = max(machine_demand.values())
        if max_demand > 0:
            for m in machine_demand:
                machine_demand[m] /= max_demand
    else:
        pass

    # Score each (i, m) pair
    best_pair_score = -np.inf
    best_i = 0
    best_m = None

    for i, op in enumerate(ready_operations):
        if not op["machine_options"]:
            continue
        op_score = score[i]
        for m, dur in op["machine_options"].items():
            demand = machine_demand.get(m, 0.0)
            # Pair score: high op score, low duration, low machine demand
            pair_score = op_score - 0.3 * (dur / max(hard_min_durs[i], 1e-6)) - 0.15 * demand
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_i = i
                best_m = m

    if best_m is not None:
        return (best_i, best_m)
    return best_i

# EVOLVE-BLOCK-END
