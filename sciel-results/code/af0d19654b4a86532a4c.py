import time as time_module
import math

# ------------------------------------------------------------------
#  Open-Shop dispatch rule (refined within the multi-criterion
#  heuristic family).
#
#  Design: composite score combining remaining work, SPT bonus,
#  machine-load balance, job urgency, and a smooth makespan
#  surrogate. A 1-step look-ahead refines the top-K candidates.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START = time_module.time()
_MARGIN = 2.0

# Tuned weights
W_MWR  = 1.0     # remaining work (higher = more work remaining)
W_SPT  = 0.15    # short-processing-time bonus
W_MACH = 0.20    # machine-load balance penalty (moderate)
W_URG  = 0.25    # urgency: more remaining ops -> higher priority
W_MK   = 1.0     # makespan surrogate penalty (lower is better)
W_SPREAD = 0.10  # penalty for completion-time spread

_LSE_TEMP = 1.5   # log-sum-exp temperature
_BEAM    = 5
_TOP_K   = 5


def _elapsed():
    return time_module.time() - _START


def _remaining():
    return _DEADLINE - _MARGIN - _elapsed()


def _lse2(a, b, temp):
    """Smooth max of two values via log-sum-exp."""
    m = a if a > b else b
    if temp <= 0:
        return m
    return m + temp * math.log(math.exp((a - m) / temp) + math.exp((b - m) / temp))


def _lse_multi(vals, temp):
    """Smooth max of a list of values via log-sum-exp."""
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    m = max(vals)
    if temp <= 0:
        return m
    s = 0.0
    for v in vals:
        s += math.exp((v - m) / temp)
    return m + temp * math.log(s)


def _count_loads(ops):
    """Count how many ready ops can use each machine."""
    loads = {}
    for op in ops:
        mo = op.get("machine_options", {})
        for m in mo:
            loads[m] = loads.get(m, 0) + 1
    return loads


def _est_machine_finish(ops, t):
    """
    Estimate when each machine will be free based on the minimum
    duration among ops that can use it. Returns {machine: est_finish}.
    """
    est = {}
    for op in ops:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        md = min(mo.values())
        for m in mo:
            est[m] = est.get(m, 0.0) + md
    for m in est:
        est[m] += t
    return est


def _est_completion_times(ops, t, mfin):
    """
    For each ready op, estimate its earliest completion time given
    current machine availability. Returns a list parallel to ops.
    """
    comps = []
    for op in ops:
        mo = op.get("machine_options", {})
        if not mo:
            comps.append(t)
            continue
        best = min(mfin.get(m, t) + mo[m] for m in mo)
        comps.append(best)
    return comps


def _score(op, loads, t, mfin):
    mo = op.get("machine_options", {})
    rw = op.get("remaining_work", 0.0)
    rem_ops = op.get("remaining_ops", [])
    n_rem = len(rem_ops) if isinstance(rem_ops, (list, tuple)) else 0

    durs = list(mo.values()) if mo else [0.0]
    min_dur = min(durs) if durs else 0.0

    # Average machine congestion among candidate machines (softer than max)
    if mo:
        avg_load = sum(loads.get(m, 0) for m in mo) / len(mo)
    else:
        avg_load = 0

    # Estimated completion on the best machine
    if mfin is not None and mo:
        best_comp = min(mfin.get(m, t) + mo[m] for m in mo)
    else:
        best_comp = t + min_dur

    # Smooth makespan surrogate: combine completion time and remaining work
    mk = _lse2(best_comp, rw, _LSE_TEMP)

    s = (W_MWR * rw
         + W_SPT * (1.0 / (min_dur + 1.0))
         - W_MACH * avg_load
         + W_URG * (n_rem / (n_rem + 1.0))
         - W_MK * mk)
    return s


def _lookahead_score(ops, idx, t, loads, mfin):
    """
    1-step look-ahead: tentatively assign op[idx] to its best machine,
    recompute completion times for remaining ops, and return a
    composite score based on the resulting makespan surrogate and
    completion-time spread.
    """
    op = ops[idx]
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0

    # Pick the machine with the earliest completion
    best_m = min(mo, key=lambda m: (mfin.get(m, t) if mfin else t) + mo[m])
    comp = (mfin.get(best_m, t) if mfin else t) + mo[best_m]

    # Update loads: remove this op from all its machines, add to best_m
    new_loads = dict(loads)
    for m in mo:
        if m in new_loads:
            new_loads[m] -= 1
    new_loads[best_m] = new_loads.get(best_m, 0) + 1

    # Update est finish for best_m
    new_mfin = dict(mfin) if mfin else {}
    new_mfin[best_m] = comp

    # Compute completion times for remaining ops under new state
    comp_times = []
    for j, other in enumerate(ops):
        if j == idx:
            continue
        mo2 = other.get("machine_options", {})
        if not mo2:
            comp_times.append(t)
            continue
        bc2 = min(new_mfin.get(m, t) + mo2[m] for m in mo2)
        comp_times.append(bc2)

    # Makespan surrogate: smooth max over all completion times
    if comp_times:
        mk_after = _lse_multi(comp_times, _LSE_TEMP)
    else:
        mk_after = comp

    # Completion-time spread penalty: reward even scheduling
    if len(comp_times) >= 2:
        mean_ct = sum(comp_times) / len(comp_times)
        spread = sum((c - mean_ct) ** 2 for c in comp_times) / len(comp_times)
        spread_penalty = W_SPREAD * spread
    else:
        spread_penalty = 0.0

    return -mk_after - spread_penalty


def _fast(ops, t):
    loads = _count_loads(ops)
    mfin = _est_machine_finish(ops, t)
    best_i = 0
    best_s = float('-inf')
    for i, op in enumerate(ops):
        s = _score(op, loads, t, mfin)
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _remaining() < 0.3:
        return _fast(ready_operations, time)

    n = len(ready_operations)
    loads = _count_loads(ready_operations)
    mfin = _est_machine_finish(ready_operations, time)

    # Score all
    scored = []
    for i, op in enumerate(ready_operations):
        scored.append((_score(op, loads, time, mfin), i))
    scored.sort(key=lambda x: x[0], reverse=True)

    # If very few ops, no need for look-ahead
    if n <= 2:
        return scored[0][1]

    # Beam search over top-K with 1-step look-ahead
    k = min(_TOP_K, n)
    best_idx = scored[0][1]
    best_combined = float('-inf')

    for r in range(k):
        base_s, idx = scored[r]
        la = _lookahead_score(ready_operations, idx, time, loads, mfin)
        combined = base_s + 0.5 * la
        if combined > best_combined:
            best_combined = combined
            best_idx = idx
        if _remaining() < 1.0:
            break

    return best_idx
