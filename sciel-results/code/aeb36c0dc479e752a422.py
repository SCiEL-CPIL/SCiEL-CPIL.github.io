"""Job-Shop Scheduling Dispatch Rule.
SPT-dominant dispatch with criticality and load-balancing penalty.
Motivation: shortest-processing-time is the dominant makespan reducer in
small (10-job, 5-machine) job-shops; long operations left unscheduled
become critical-path bottlenecks. We rank primarily by min duration
(SPT), tie-break toward high criticality, and penalize operations whose
chosen machine already carries the most remaining work (load balancing).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 120s/150s limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Deadline guard: fall back to pure SPT (min min-duration) with best machine
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = np.empty(n, dtype=np.float64)
        best_m = [None] * n
        for i, op in enumerate(ready_operations):
            mo = op.get("machine_options", {})
            if mo:
                bm, bd = min(mo.items(), key=lambda kv: kv[1])
                min_durs[i] = bd
                best_m[i] = bm
            else:
                min_durs[i] = float("inf")
        idx = int(np.argmin(min_durs))
        if best_m[idx] is not None:
            return (idx, best_m[idx])
        return idx

    # ---- Extract per-operation features (loop is cheap: n <= 10) ----
    min_dur = np.zeros(n, dtype=np.float64)      # hard min processing time (SPT key)
    best_mach = [None] * n
    remaining_work = np.zeros(n, dtype=np.float64)
    remaining_ops = np.ones(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            bm, bd = min(mo.items(), key=lambda kv: kv[1])
            min_dur[i] = float(bd)
            best_mach[i] = bm
        else:
            min_dur[i] = float("inf")
        remaining_work[i] = float(op.get("remaining_work", 0.0))
        remaining_ops[i] = float(op.get("remaining_ops", 1))

    # Safety: any inf duration (no machine) should be last-resort
    finite = np.isfinite(min_dur)
    if not finite.all():
        # If some ops have no machine, prefer those that do (finite)
        if finite.any():
            # mask out infeasible
            min_dur = np.where(finite, min_dur, np.inf)

    # ---- Load proxy: total remaining work per machine ----
    # Build machine -> sum of min_dur over ops that would use it, as a proxy
    # for machine load. This penalizes dispatching onto a congested machine.
    machine_load = {}
    for i in range(n):
        m = best_mach[i]
        if m is not None:
            machine_load[m] = machine_load.get(m, 0.0) + min_dur[i]
    # Also fold in each op's own remaining_work to reflect job-side load
    load_penalty = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m = best_mach[i]
        if m is not None:
            load_penalty[i] = machine_load.get(m, 0.0)

    # ---- Build normalized scores ----
    # SPT: shorter min_dur is better
    md_finite = min_dur[np.isfinite(min_dur)]
    md_min = np.min(md_finite) if md_finite.size else 0.0
    md_max = np.max(md_finite) if md_finite.size else 1.0
    if md_max > md_min:
        spt = (md_max - min_dur) / (md_max - md_min)          # 1 = shortest
    else:
        spt = np.ones(n, dtype=np.float64)

    # Criticality: higher remaining_work is more critical
    rw_max = np.max(remaining_work) if np.max(remaining_work) > 0 else 1.0
    criticality = remaining_work / rw_max

    # Urgency: fewer remaining ops is closer to completion
    ro_safe = np.maximum(remaining_ops, 1.0)
    ro_max = np.max(ro_safe) if np.max(ro_safe) > 0 else 1.0
    urgency = (ro_max - ro_safe) / ro_max

    # Load-balance: lower machine load is better (avoid congestion)
    lp_finite = load_penalty[np.isfinite(load_penalty)]
    lp_max = np.max(lp_finite) if lp_finite.size and np.max(lp_finite) > 0 else 1.0
    load = 1.0 - (load_penalty / lp_max) if lp_max > 0 else np.ones(n, dtype=np.float64)

    # ---- Weighted composite: SPT dominant ----
    w_spt = 1.5
    w_crit = 0.4
    w_urg = 0.3
    w_load = 0.3

    scores = (w_spt * spt) + (w_crit * criticality) + (w_urg * urgency) + (w_load * load)

    # Tie-break / stability: among near-equal scores, prefer shorter min_dur
    best_idx = int(np.argmax(scores))

    bm = best_mach[best_idx]
    if bm is not None:
        return (best_idx, bm)
    return best_idx

# EVOLVE-BLOCK-END
