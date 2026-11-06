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

    # Machine-contention avoidance scoring:
    # For each ready operation, compute a "contention score" that reflects
    # how much this operation is bottlenecked by machine load.
    #
    # Key idea: an operation with a short min-duration relative to the total
    # duration across its available machines is less dependent on any single
    # machine being free, so it can be scheduled more flexibly. Conversely,
    # an operation whose duration is concentrated on one heavily-loaded machine
    # is more likely to cause or suffer from idle time.
    #
    # Score per operation i:
    #   total_dur_i = sum of durations across all machine options for op i
    #   min_dur_i   = min duration across machine options for op i
    #   flexibility_i = min_dur_i / (total_dur_i + 1e-9)   # in (0, 1]
    #
    #   machine_load[m] = sum of durations of all ready ops that can use machine m
    #   min_load_i = min machine_load[m] for m in op i's options
    #
    #   # Remaining work and ops give urgency
    #   rw_i = op["remaining_work"]
    #   ro_i = op["remaining_ops"]
    #   urgency_i = rw_i / (ro_i + 1)  # average remaining work per op
    #
    #   # Combined score: prefer high urgency, high flexibility, low min_load
    #   score_i = urgency_i * flexibility_i / (min_load_i + 1)
    #
    # This smooths the hard min via the ratio, avoiding the nonsmoothness
    # of picking the absolute minimum machine.

    # Precompute machine load from the ready queue
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0) + d

    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        vals = list(opts.values())
        min_dur = min(vals)
        total_dur = sum(vals)

        # Flexibility: how spread out the duration is across machines
        # Higher = more flexible (min is a larger fraction of total)
        flexibility = min_dur / (total_dur + 1e-9)

        # Minimum load among available machines
        min_load = min(machine_load[m] for m in opts)

        # Urgency: average remaining work per remaining op
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        urgency = rw / (ro + 1) if ro > 0 else rw

        # Composite score
        score = urgency * flexibility / (min_load + 1)

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
