"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE_SECONDS = 115.0

# Track machine load across dispatch calls to detect contention
_machine_load = {}  # machine -> total duration assigned so far
_job_remaining = {}  # job -> remaining work (updated as we dispatch)

def _check_deadline():
    return _time.time() - _start_time < _DEADLINE_SECONDS

def _soft_min(values, temperature=0.5):
    """Soft-min via log-sum-exp: smooth surrogate for min()."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    m = arr.min()
    if m == 0:
        return 0.0
    exp_vals = np.exp(-(arr - m) / temperature)
    s = exp_vals.sum()
    if s == 0:
        return m
    return m - temperature * np.log(s)

def _soft_max(values, temperature=0.5):
    """Soft-max via log-sum-exp: smooth surrogate for max()."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    m = arr.max()
    if m == 0:
        return 0.0
    exp_vals = np.exp((arr - m) / temperature)
    s = exp_vals.sum()
    if s == 0:
        return m
    return m + temperature * np.log(s)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if not _check_deadline():
        return 0

    n = len(ready_operations)
    scores = np.zeros(n, dtype=np.float64)

    # Pre-compute machine load statistics
    all_machines = set()
    for op in ready_operations:
        for m in op["machine_options"]:
            all_machines.add(m)

    if all_machines:
        loads = np.array([_machine_load.get(m, 0.0) for m in all_machines])
        load_mean = loads.mean() if len(loads) > 0 else 0.0
        load_std = loads.std() if len(loads) > 1 else 1.0
        if load_std < 1e-9:
            load_std = 1.0
    else:
        load_mean = 0.0
        load_std = 1.0

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op.get("remaining_ops", 0)
        durations = list(op["machine_options"].values())
        machines = list(op["machine_options"].keys())

        if not durations:
            d_min = 0.0
            d_max = 0.0
            d_mean = 0.0
            d_soft_min = 0.0
            d_soft_max = 0.0
            n_options = 1
        else:
            d_min = float(min(durations))
            d_max = float(max(durations))
            d_mean = float(np.mean(durations))
            d_soft_min = _soft_min(durations, temperature=0.3)
            d_soft_max = _soft_max(durations, temperature=0.3)
            n_options = len(durations)

        # Machine contention: how loaded are the candidate machines?
        # Lower load = less contention = better
        if machines:
            machine_loads = np.array([_machine_load.get(m, 0.0) for m in machines])
            # Normalize: lower is better, so we subtract from mean
            contention = float(np.mean(machine_loads))
            # Soft-min of machine loads: if any machine is free, it's less constraining
            min_machine_load = float(np.min(machine_loads))
        else:
            contention = 0.0
            min_machine_load = 0.0

        # Critical path estimate: remaining ops * max duration (worst case per op)
        # Use soft-max for smoothness
        critical_path = ro * d_soft_max if ro > 0 else d_soft_max

        # Job urgency: total remaining work for this job
        # Jobs with more remaining work are more urgent
        job_urgency = rw

        # Flexibility: more machine options = less constraining = can be scheduled later
        # So we penalize operations with fewer options (they're more urgent)
        flexibility_penalty = 1.0 / (1.0 + n_options)

        # Duration spread: if min << max, the operation is flexible in duration choice
        duration_spread = d_max - d_min

        # Composite score - higher = dispatch first
        # 1. Critical path dominance: operations on the critical path first
        # 2. Job remaining work: big jobs first (LPT)
        # 3. Machine contention: avoid overloaded machines
        # 4. Immediate processing time: longer ops first to reduce tail
        # 5. Flexibility: less flexible ops first (fewer options)
        score = (1.2 * critical_path          # critical path
                 + 0.8 * job_urgency           # LPT: remaining work
                 - 0.5 * (contention / (load_mean + 1e-9))  # avoid loaded machines
                 + 0.3 * d_soft_min            # immediate LPT
                 + 0.2 * flexibility_penalty   # less flexible first
                 - 0.1 * (duration_spread / (d_mean + 1e-9)))  # spread penalty

        scores[i] = score

    # Find the best operation
    best_idx = int(np.argmax(scores))

    # Update machine load tracking for the chosen operation
    # We need to pick a specific machine - choose the least loaded one
    op = ready_operations[best_idx]
    if op["machine_options"]:
        # Choose the machine with minimum current load
        min_load = float('inf')
        chosen_machine = None
        for m, dur in op["machine_options"].items():
            ml = _machine_load.get(m, 0.0)
            if ml < min_load:
                min_load = ml
                chosen_machine = m
        if chosen_machine is not None:
            _machine_load[chosen_machine] = _machine_load.get(chosen_machine, 0.0) + op["machine_options"][chosen_machine]

    return best_idx
# EVOLVE-BLOCK-END
