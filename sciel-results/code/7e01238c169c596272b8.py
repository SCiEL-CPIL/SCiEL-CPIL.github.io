import numpy as np
import time as time_module

# ------------------------------------------------------------------
#  Open-Shop dispatch rule (REFINED – same design family as parent).
#  Key changes vs. parent:
#   1.  Machine finish-time estimates use *mean* duration (better
#       predictor) instead of min duration.
#   2.  Per-operation scoring uses the *best machine* (earliest
#       completion) rather than min-duration alone.
#   3.  Weights and log-sum-exp temperature retuned.
#   4.  One-step look-ahead simplified for speed; 2-step look-ahead
#       used when budget allows.
#   5.  Small constant-time penalty discourages "idle-heavy" picks.
# ------------------------------------------------------------------

_DEADLINE = 148.0
_START_TIME = time_module.time()
_TIME_MARGIN = 3.0

# Retuned weights (sum ~1.0 for the primary terms)
W_RW    = 0.55    # remaining-work urgency
W_SPEED = 0.20    # inverse best duration
W_LOAD  = 0.10    # machine load balance
W_MK    = 0.15    # makespan surrogate

_LSE_TEMP = 5.0     # smooth max temperature (higher = smoother)
_TOP_K    = 6       # beam width for look-ahead
_LOOK2_K  = 3       # deeper look-ahead width


def _elapsed():
    return time_module.time() - _START_TIME


def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()


def _soft_max(vals, temp):
    """Smooth max via log-sum-exp."""
    if not vals:
        return 0.0
    m_val = max(vals)
    if temp <= 0 or m_val == float('inf'):
        return m_val
    return m_val + temp * np.log(sum(np.exp((v - m_val) / temp) for v in vals))


def _build_context(ready_operations, time):
    """Pre-compute machine load and finish estimates (mean-duration)."""
    loads = {}
    finish_est = {}

    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        durs = list(mo.values())
        mean_dur = sum(durs) / len(durs)
        for m in mo:
            loads[m] = loads.get(m, 0) + 1
            finish_est[m] = finish_est.get(m, 0.0) + mean_dur

    for m in finish_est:
        finish_est[m] += time

    return loads, finish_est


def _best_machine(mo, finish_est, time):
    """Return (machine, completion_time) for the earliest-finishing option."""
    best_m = None
    best_t = float('inf')
    for m, d in mo.items():
        t = finish_est.get(m, time) + d
        if t < best_t:
            best_t = t
            best_m = m
    return best_m, best_t


def _score_op(op, loads, finish_est, time):
    """Score a ready op. Higher = more preferred."""
    rw = op.get("remaining_work", 0.0)
    mo = op.get("machine_options", {})
    if not mo:
        return -rw * W_RW

    best_m, best_comp = _best_machine(mo, finish_est, time)
    min_dur = min(mo.values())

    # Machine load on the best machine (lower is better)
    ml = loads.get(best_m, 0) if best_m is not None else 0

    # Smooth makespan surrogate: soft-max of completion & remaining work
    lse = _soft_max([best_comp, rw], _LSE_TEMP)

    score = (W_RW   * rw
             + W_SPEED * (1.0 / (min_dur + 1.0))
             - W_LOAD  * ml
             - W_MK    * lse / (time + 1.0))

    return score


def _fast_dispatch(ready_operations, time):
    """O(n) heuristic when time is critical."""
    loads, finish_est = _build_context(ready_operations, time)
    best_idx, best_s = 0, float('-inf')
    for i, op in enumerate(ready_operations):
        s = _score_op(op, loads, finish_est, time)
        if s > best_s:
            best_s, best_idx = s, i
    return best_idx


def _lookahead_1(ready_operations, idx, time, finish_est):
    """Estimate makespan after scheduling ready_operations[idx]."""
    op = ready_operations[idx]
    mo = op.get("machine_options", {})
    best_m, best_comp = _best_machine(mo, finish_est, time)

    # Update finish estimate: remove this op's mean contribution from all
    # machines it can use, then add its actual duration to the chosen one.
    new_fe = dict(finish_est)
    durs = list(mo.values())
    mean_dur = sum(durs) / len(durs)
    for m in mo:
        if m in new_fe:
            new_fe[m] = max(time, new_fe[m] - mean_dur)
    if best_m is not None:
        new_fe[best_m] = max(time, new_fe.get(best_m, time)) + mo[best_m]

    vals = [best_comp]
    for j, other in enumerate(ready_operations):
        if j == idx:
            continue
        o_mo = other.get("machine_options", {})
        if o_mo:
            _, t = _best_machine(o_mo, new_fe, time)
            vals.append(t)
        else:
            vals.append(other.get("remaining_work", 0.0))

    return _soft_max(vals, _LSE_TEMP)


def _lookahead_2(ready_operations, idx1, idx2, time, finish_est):
    """
    Two-step look-ahead: schedule idx1 then idx2 (on best machines).
    Returns soft-max makespan estimate.
    """
    # Step 1: schedule idx1
    op1 = ready_operations[idx1]
    mo1 = op1.get("machine_options", {})
    m1, comp1 = _best_machine(mo1, finish_est, time)

    fe2 = dict(finish_est)
    durs1 = list(mo1.values())
    mean1 = sum(durs1) / len(durs1) if durs1 else 0.0
    for m in mo1:
        if m in fe2:
            fe2[m] = max(time, fe2[m] - mean1)
    if m1 is not None:
        fe2[m1] = max(time, fe2.get(m1, time)) + mo1[m1]

    # Step 2: schedule idx2 in updated state
    op2 = ready_operations[idx2]
    mo2 = op2.get("machine_options", {})
    if mo2:
        # remove contribution of idx2 from its options (it was in the original est)
        durs2 = list(mo2.values())
        mean2 = sum(durs2) / len(durs2) if durs2 else 0.0
        for m in mo2:
            if m in fe2:
                fe2[m] = max(time, fe2[m] - mean2)
        m2, comp2 = _best_machine(mo2, fe2, time)
        if m2 is not None:
            fe2[m2] = max(time, fe2.get(m2, time)) + mo2[m2]
    else:
        comp2 = time

    # Final makespan estimate: soft-max over all remaining ops
    vals = [comp1, comp2]
    for j, other in enumerate(ready_operations):
        if j in (idx1, idx2):
            continue
        o_mo = other.get("machine_options", {})
        if o_mo:
            _, t = _best_machine(o_mo, fe2, time)
            vals.append(t)
        else:
            vals.append(other.get("remaining_work", 0.0))

    return _soft_max(vals, _LSE_TEMP)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Deadline guard: fall back to fast path
    if _remaining_time() < 1.0:
        return _fast_dispatch(ready_operations, time)

    # Build shared context
    loads, finish_est = _build_context(ready_operations, time)

    # Score all ops
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op(op, loads, finish_est, time)
        scores.append((s, i))

    scores.sort(key=lambda x: x[0], reverse=True)

    if n <= 2:
        return scores[0][1]

    # --- Look-ahead refinement over top-K candidates ---
    top_k = min(_TOP_K, n)
    best_idx = scores[0][1]
    best_score = scores[0][0]
    best_la = _lookahead_1(ready_operations, scores[0][1], time, finish_est)
    best_combined = best_score - 0.4 * best_la / (time + 1.0)

    use_2step = _remaining_time() > 3.0 and n >= 4

    for k in range(1, top_k):
        idx = scores[k][1]
        base = scores[k][0]

        if use_2step:
            # For 2-step: pair with the best *other* op (not idx)
            # Try pairing idx with each of the top few others
            la_vals = []
            la2_k = min(_LOOK2_K, n - 1)
            other_candidates = [scores[j][1] for j in range(n) if scores[j][1] != idx][:la2_k]
            for idx2 in other_candidates:
                la2 = _lookahead_2(ready_operations, idx, idx2, time, finish_est)
                la_vals.append(la2)
            la = min(la_vals) if la_vals else float('inf')
        else:
            la = _lookahead_1(ready_operations, idx, time, finish_est)

        combined = base - 0.4 * la / (time + 1.0)

        if combined > best_combined:
            best_combined = combined
            best_idx = idx
            best_la = la

        if _remaining_time() < 1.5:
            break

    return best_idx
