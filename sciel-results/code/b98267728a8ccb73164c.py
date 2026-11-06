"""Open-Shop scheduling dispatch rule (model-and-optimize approach)."""
# EVOLVE-BLOCK-START
import time as _time

# ---- deadline guard -------------------------------------------------------
_T0 = _time.time()
_BUDGET = 140.0          # evaluator kills at 150s; keep 10s safety margin
_DEADLINE = _T0 + _BUDGET

# ---- tunable model --------------------------------------------------------
# score = w_dur * duration + w_job * remaining_work + w_load * machine_load_penalty
# We fit weights offline against representative instances.
_WEIGHTS = (1.0, 0.0, 0.0)  # default: pure LPT (duration) until tuned


def _make_instance(n_jobs=10, n_machines=5, seed=0):
    """Deterministic random open-shop instance: each job has all ops."""
    import random
    rng = random.Random(seed)
    durations = [[rng.randint(1, 20) for _ in range(n_machines)] for _ in range(n_jobs)]
    return durations


def _simulate(durations, weights):
    """Event-driven open-shop simulation with weighted dispatch rule.
    Returns makespan (C_max)."""
    import heapq
    n_jobs, n_machines = len(durations), len(durations[0])
    w_dur, w_job, w_load = weights

    remaining_ops = [set(range(n_machines)) for _ in range(n_jobs)]
    remaining_work = [sum(durations[j]) for j in range(n_jobs)]
    machine_free = [0.0] * n_machines
    job_free = [0.0] * n_jobs

    # All operations are initially ready in open-shop
    ready = [(j, m) for j in range(n_jobs) for m in range(n_machines)]
    total_ops = n_jobs * n_machines
    done = 0
    makespan = 0.0

    while done < total_ops:
        # Find the best operation to start next
        best_key = None
        best_op = None
        best_start_t = None
        
        for (j, m) in ready:
            start_t = max(machine_free[m], job_free[j])
            dur = durations[j][m]
            # Machine load: how busy is this machine relative to others
            machine_load = machine_free[m]
            # Score: prefer shorter durations, jobs with more remaining work,
            # and less loaded machines
            score = w_dur * dur + w_job * remaining_work[j] - w_load * machine_load
            # Primary: start as early as possible; Secondary: highest score
            key = (start_t, -score)
            if best_key is None or key < best_key:
                best_key = key
                best_op = (j, m)
                best_start_t = start_t
        
        if best_op is None:
            break
            
        j, m = best_op
        ready.remove((j, m))
        end_t = best_start_t + durations[j][m]
        machine_free[m] = end_t
        job_free[j] = end_t
        remaining_ops[j].discard(m)
        remaining_work[j] -= durations[j][m]
        makespan = max(makespan, end_t)
        done += 1
    
    return makespan


def _tune_weights():
    """Fit (w_dur, w_job, w_load) on representative instances. Returns tuple."""
    global _WEIGHTS
    grid = [
        (1.0, 0.0, 0.0),   # LPT
        (1.0, 0.5, 0.0),
        (1.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.8, 0.5, 0.5),
        (1.2, 0.3, 0.3),
        (1.0, 0.8, 0.2),
        (1.5, 0.0, 0.5),
    ]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    instances = [_make_instance(seed=s) for s in seeds]
    best_tuple = (1.0, 0.0, 0.0)
    best_obj = float('inf')
    for tup in grid:
        total = 0.0
        for inst in instances:
            total += _simulate(inst, tup)
        obj = total / len(instances)
        if obj < best_obj:
            best_obj = obj
            best_tuple = tup
        if _time.time() > _DEADLINE - 25.0:
            break
    _WEIGHTS = best_tuple
    return best_tuple


# Run tuning once at import time (fast; guarded by deadline).
_tune_weights()


def dispatch(ready_operations, time):
    w_dur, w_job, w_load = _WEIGHTS
    best_idx = 0
    best_score = -float('inf')
    n = len(ready_operations)
    
    # Estimate machine loads from the current time context
    # We don't have direct access to machine free times, but we can use
    # a heuristic: operations on machines with fewer pending ops are less loaded
    
    # Count how many ready ops target each machine as a proxy for load
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Choose the machine option with minimum duration (fastest completion)
        min_dur = min(opts.values())
        # Find which machine has the minimum duration
        min_machines = [m for m, d in opts.items() if d == min_dur]
        # Among min-duration machines, pick the one with lowest estimated load
        if len(min_machines) > 1:
            chosen_m = min(min_machines, key=lambda m: machine_count.get(m, 0))
        else:
            chosen_m = min_machines[0]
        
        dur = opts[chosen_m]
        machine_load_proxy = machine_count.get(chosen_m, 0)
        
        score = w_dur * dur + w_job * op["remaining_work"] - w_load * machine_load_proxy
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
