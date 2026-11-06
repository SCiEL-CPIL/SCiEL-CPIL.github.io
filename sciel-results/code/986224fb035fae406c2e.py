"""Load-Balancing Water-Filling dispatch for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = 140.0          # safety margin before the 150 s kill
_START = _time.monotonic()


def dispatch(ready_operations, time):
    # Hard deadline guard: if we are close to the kill, fall back fast.
    if _time.monotonic() - _START > _DEADLINE:
        return int(max(range(len(ready_operations)),
                       key=lambda i: ready_operations[i]["remaining_work"]))

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Collect the set of all machines that appear in any candidate.
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    mlist = sorted(all_machines)
    m_index = {m: i for i, m in enumerate(mlist)}
    n_m = len(mlist)

    # Current machine load is not directly given; we approximate the
    # *additional* load each operation would add. To do water-filling we
    # need a notion of current load. We infer it from the fact that the
    # simulator only hands us *ready* operations (their machine is free
    # right now). So at this decision instant, the machine load relevant
    # to *future* congestion is the sum of durations of operations that
    # are still in the pipeline (remaining_work per job is a proxy for
    # that job's future demand, but machine load is the sum over jobs
    # that will use that machine).
    #
    # Since we don't have a global machine-load state, we use a
    # *local* water-filling: for each candidate operation, the best
    # machine is the one with the smallest duration (shortest processing
    # time on that machine), which minimizes the time the machine is
    # blocked. We then score the operation by a combination of:
    #   (a) the duration on its best (least-blocking) machine  -> lower better
    #   (b) the job's remaining_work                           -> higher better
    #   (c) the job's remaining_ops                            -> higher better
    #
    # Water-filling interpretation: pick the operation whose execution
    # on its shortest machine causes the smallest "peak" increase, while
    # still making progress on jobs with lots of work left (so we don't
    # starve a long job and let its tail dominate the makespan).

    best_score = -math.inf
    best_idx = 0
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Shortest processing time among available machines.
        spt = min(opts.values())
        # Number of machines offering the SPT (tie-break: more flexibility is better).
        n_spt = sum(1 for v in opts.values() if v == spt)
        # Mean duration (a proxy for how "expensive" this op is overall).
        mean_dur = sum(opts.values()) / len(opts)
        # Max duration (worst-case blocking if we picked badly).
        max_dur = max(opts.values())

        rw = op["remaining_work"]
        ro = op["remaining_ops"]

        # Water-filling score: we want to MINIMIZE the resulting peak load.
        # Using the SPT machine, the incremental peak is spt. We combine:
        #   primary: minimize spt (keep machines free quickly)
        #   secondary: maximize rw (clear heavy jobs)
        #   tertiary: maximize ro (finish jobs)
        #   quaternary: prefer flexibility (n_spt) and lower mean/max
        #
        # Scale so that spt (in time units) dominates, then rw/ro nudge.
        # We use a lexicographic-ish blend with small weights.
        score = -spt + 0.01 * rw + 0.001 * ro + 0.0001 * n_spt - 0.0001 * mean_dur

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
