"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()

    # Deadline guard: evaluator kills at 150s; use 145s as safety margin.
    if _time.time() - _START_TIME > 145:
        # Fast fallback: shortest processing time
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op["machine_options"].values())
            if d < best_dur:
                best_dur = d
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Precompute machine load from the ready queue for contention awareness
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0) + d

    total_load = sum(machine_load.values()) + n_ops  # normalization factor

    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        vals = list(opts.values())
        min_dur = min(vals)
        total_dur = sum(vals)

        # Soft-min via log-sum-exp: smoother surrogate for effective duration
        max_val = max(vals)
        lse = max_val + _math.log(sum(_math.exp(v - max_val) for v in vals))
        soft_min = lse  # log-sum-exp of durations

        # Flexibility: how spread out the duration is across machines
        flexibility = min_dur / (total_dur + 1e-9)

        # Minimum load among available machines (contention avoidance)
        min_load = min(machine_load[m] for m in opts)

        # Urgency: critical path slack
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        urgency = rw / (ro + 1) if ro > 0 else rw

        # Critical path slack: prioritize jobs with high remaining work
        # relative to the time it takes to complete the current operation
        critical_slack = rw / (soft_min + 1e-9)

        # Load penalty: how loaded is the best machine relative to total load
        best_machine = min(opts.keys(), key=lambda m: opts[m])
        load_penalty = machine_load[best_machine] / total_load

        # Machine-contention avoidance: inverse of utilization ratio
        # Prioritize operations that can run on less loaded machines
        # Using the ratio of min duration to total duration as a proxy for
        # how constrained the operation is
        contention_score = min_dur / (total_dur + 1e-9)
        
        # Combine: high urgency, good flexibility, low contention, low load
        # The contention_score rewards operations that are less dependent
        # on a single machine (higher flexibility) and can run on less loaded machines
        score = critical_slack * flexibility * (1 - load_penalty) / (min_load + 1)
        
        # Add machine-contention avoidance term
        # Operations with higher contention_score (more flexible) get a boost
        # This reduces idle time by spreading work across less loaded machines
        score += 0.1 * contention_score * rw / (total_load + 1)

        # Add small tie-breaker based on remaining work
        score += 1e-6 * rw

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
