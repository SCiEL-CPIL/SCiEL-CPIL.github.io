"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = _time.monotonic() + 110  # safety margin under the 120s wall-clock limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we are running long, fall back to a fast, safe choice.
    if _time.monotonic() > _DEADLINE:
        return 0

    # Per-operation features.
    # For each ready op we know machine_options {machine: duration}.
    # We compute:
    #   min_dur  = shortest processing time among its machines (SPT preference)
    #   min_load = among its machines, the one with the least total assigned
    #              work (load balancing to avoid bottleneck machines)
    #   load_of_min = the total work already assigned to that least-loaded machine
    #   remaining_work from the dict (job-level remaining work, MRW tie-break)

    # Aggregate machine loads across the ready set: for each machine, sum the
    # durations of ready operations that could use it. This is a proxy for how
    # congested each machine will be in the near term.
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            machine_load[m] = machine_load.get(m, 0) + d

    best_idx = 0
    best_key = None

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            continue

        # SPT: prefer the operation with the shortest processing time.
        min_dur = min(mo.values())

        # Among the machines that can process this op, pick the least loaded one.
        # This avoids piling more work onto an already-busy machine.
        min_load = min(machine_load.get(m, 0) for m in mo)

        # Remaining work of the job as a tie-breaker (longer remaining job first).
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)

        # Composite key: we want to MINIMIZE (min_dur, min_load) and MAXIMIZE rw.
        # Build a tuple that is comparable; we pick the op with the best (smallest)
        # (min_dur, min_load) and among ties the largest rw.
        key = (min_dur, min_load, -rw, -ro)

        if best_key is None or key < best_key:
            best_key = key
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
