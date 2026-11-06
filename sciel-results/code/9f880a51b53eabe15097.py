import numpy as np
import time as time_module

# ------------------------------------------------------------------
#  Open-Shop dispatch rule.
#
#  Strategy: multi-criterion heuristic scoring with a smooth
#  log-sum-exp surrogate for the makespan (max-completion-time)
#  objective, combined with machine-finish-time estimation for
#  smarter machine selection.  A small top-K beam with one-step
#  look-ahead refines the top candidates.  A deadline guard ensures
#  we never exceed the wall-clock budget.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0

# Weights
W_MWR  = 0.7    # remaining work (job-tail)
W_SPT  = 0.1    # inverse min duration (speed)
W_MACH = 0.05   # machine load balance
W_MK   = 0.5    # makespan surrogate weight (lower is better)

_TOP_K = 5
_LSE_TEMP = 3.0  # temperature for log-sum-exp (lower = sharper max)


def _elapsed():
    return time_module.time() - _START_TIME


def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()


def _soft_max(vals, temp):
    """Smooth max via log-sum-exp. Lower temp -> closer to hard max."""
    if not vals:
        return 0.0
    m_val = max(vals)
    if temp <= 0:
        return m_val
    # Guard: if all values are very negative, exp underflows gracefully
    return m_val + temp * np.log(sum(np.exp((v - m_val) / temp) for v in vals))


def _estimate_machine_finish(ready_operations, time):
    """
    Estimate when each machine becomes free, assuming ready ops are
    processed.  For each machine, sum the minimum durations of all
    ready ops that can use it (a lower-bound busy-time estimate).
    """
    est = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        # Use the minimum duration as a conservative per-op contribution
        min_dur = min(mo.values()) if mo else 0.0
        for m in mo.keys():
            est[m] = est.get(m, 0.0) + min_dur

    for m in est:
        est[m] = time + est[m]
    return est


def _count_machine_loads(ready_operations):
    """
    Build a dict mapping machine -> count of ready ops that can use it.
    Proxy for machine congestion.
    """
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            loads[m] = loads.get(m, 0) + 1
    return loads


def _score_op(op, machine_loads, machine_est_finish, time):
    """
    Score a single ready operation.  Higher = more preferred.
    The makespan surrogate (log-sum-exp of completion and remaining
    work) is the primary driver; heuristic terms break ties.
    """
    rw = op.get("remaining_work", 0.0)
    mo = op.get("machine_options", {})

    durs = list(mo.values()) if mo else [0.0]
    min_dur = min(durs) if durs else 0.0

    # Machine load penalty: prefer least-loaded machine among options
    mach_penalty = 0.0
    if machine_loads is not None and mo:
        loads = [machine_loads.get(m, 0) for m in mo.keys()]
        min_load = min(loads) if loads else 0
        mach_penalty = min_load

    # Estimate completion time of this op on its best machine
    # (earliest finish among available machines)
    if machine_est_finish is not None and mo:
        best_finish = min(
            machine_est_finish.get(m, time) + mo[m] for m in mo
        )
    else:
        best_finish = time + min_dur

    # Smooth makespan surrogate: soft max of (completion, remaining work)
    # This captures the bottleneck: the makespan is driven by the
    # later of "when this op finishes" and "how much work remains".
    lse = _soft_max([best_finish, rw], _LSE_TEMP)

    score = (W_MWR * rw
             + W_SPT * (1.0 / (min_dur + 1.0))
             - W_MACH * mach_penalty
             - W_MK * lse)

    return score


def _simulate_one_step(ready_operations, chosen_idx, time,
                       machine_est_finish, machine_loads):
    """
    One-step look-ahead: estimate the makespan impact of starting
    ready_operations[chosen_idx] on its best machine.
    Returns a smooth makespan estimate (lower is better).
    """
    op = ready_operations[chosen_idx]
    mo = op.get("machine_options", {})
    durs = list(mo.values()) if mo else [0.0]

    # Pick the machine minimising estimated completion time
    best_machine = None
    best_comp = float('inf')
    for m in mo.keys():
        start = machine_est_finish.get(m, time) if machine_est_finish else time
        comp = start + mo[m]
        if comp < best_comp:
            best_comp = comp
            best_machine = m

    comp_time = best_comp

    # After removing this op, re-estimate machine finish times
    # (subtract this op's min-dur contribution from each machine it could use)
    new_est = dict(machine_est_finish) if machine_est_finish else {}
    if mo:
        min_dur = min(durs)
        for m in mo.keys():
            if m in new_est:
                new_est[m] -= min_dur
    # Add back on the chosen machine (it's now committed)
    if best_machine is not None and mo:
        new_est[best_machine] = new_est.get(best_machine, time) + mo[best_machine]

    # Estimate remaining makespan: soft-max over
    #   - completion time of chosen op
    #   - for each other ready op, its estimated completion on best machine
    vals = [comp_time]
    for j, other in enumerate(ready_operations):
        if j == chosen_idx:
            continue
        o_mo = other.get("machine_options", {})
        if o_mo:
            o_best = min(
                new_est.get(m, time) + o_mo[m] for m in o_mo
            )
            vals.append(o_best)
        else:
            vals.append(other.get("remaining_work", 0.0))

    # Also include raw remaining work of others as a floor
    for j, other in enumerate(ready_operations):
        if j == chosen_idx:
            continue
        rw = other.get("remaining_work", 0.0)
        if rw > 0:
            vals.append(rw)

    lse = _soft_max(vals, _LSE_TEMP)
    return lse


def _fast_heuristic(ready_operations, time):
    """
    Fast single-pass heuristic for deadline-critical situations.
    """
    best_idx = 0
    best_score = float('-inf')

    # Quick machine load count
    loads = {}
    for op in ready_operations:
        for m in op.get("machine_options", {}):
            loads[m] = loads.get(m, 0) + 1

    # Quick machine finish estimate
    est = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        min_dur = min(mo.values())
        for m in mo.keys():
            est[m] = est.get(m, 0.0) + min_dur
    for m in est:
        est[m] = time + est[m]

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        durs = list(mo.values()) if mo else [0.0]
        min_dur = min(durs) if durs else 0.0

        min_load = min((loads.get(m, 0) for m in mo), default=0)

        if est and mo:
            best_finish = min(est.get(m, time) + mo[m] for m in mo)
        else:
            best_finish = time + min_dur

        lse = _soft_max([best_finish, rw], _LSE_TEMP)

        s = (W_MWR * rw
             + W_SPT * (1.0 / (min_dur + 1.0))
             - W_MACH * min_load
             - W_MK * lse)
        if s > best_score:
            best_score = s
            best_idx = i

    return best_idx


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    if _remaining_time() < 0.5:
        return _fast_heuristic(ready_operations, time)

    n = len(ready_operations)

    # Pre-compute shared context
    machine_loads = _count_machine_loads(ready_operations)
    machine_est_finish = _estimate_machine_finish(ready_operations, time)

    # --- Score all ops ---
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op(op, machine_loads, machine_est_finish, time)
        scores.append((s, i))

    # Sort descending by score
    scores.sort(reverse=True, key=lambda x: x[0])

    # For small problem sizes, the top score is reliable enough
    if n <= 3:
        return scores[0][1]

    # --- Beam search over top-K candidates with one-step look-ahead ---
    top_k = min(_TOP_K, n)
    best_idx = scores[0][1]
    best_combined = scores[0][0]

    for k in range(top_k):
        idx = scores[k][1]
        est_mk = _simulate_one_step(
            ready_operations, idx, time,
            machine_est_finish, machine_loads
        )

        # Composite: base heuristic score minus makespan penalty
        base = scores[k][0]
        # Normalise est_mk relative to time scale
        norm_mk = est_mk / (time + 1.0) if time > 0 else est_mk
        combined = base - 0.3 * norm_mk

        if combined > best_combined:
            best_combined = combined
            best_idx = idx

        # Deadline check inside loop
        if _remaining_time() < 1.0:
            break

    return best_idx
